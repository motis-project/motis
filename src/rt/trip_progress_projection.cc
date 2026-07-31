#include "motis/rt/trip_progress_projection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "geo/polyline.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/rt/frun.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis {
namespace {

struct cached_run_shape {
  std::vector<geo::latlng> points_;
  std::vector<double> point_distances_;
  std::vector<std::size_t> stop_point_indices_;
  std::vector<double> stop_distances_;
  std::vector<unsigned> static_stop_sequences_;
};

struct cache_key {
  n::shapes_storage const* storage_;
  n::trip_idx_t trip_;
  n::stop_idx_t from_;
  n::stop_idx_t to_;

  bool operator<(cache_key const& other) const {
    return std::tuple{reinterpret_cast<std::uintptr_t>(storage_),
                      cista::to_idx(trip_), cista::to_idx(from_),
                      cista::to_idx(to_)} <
           std::tuple{reinterpret_cast<std::uintptr_t>(other.storage_),
                      cista::to_idx(other.trip_), cista::to_idx(other.from_),
                      cista::to_idx(other.to_)};
  }
};

std::optional<n::interval<n::stop_idx_t>> get_trip_range(
    n::rt::frun const& fr, n::trip_idx_t const trip) {
  for (auto const& [transport, range] : fr.tt_->trip_transport_ranges_[trip]) {
    if (transport == fr.t_.t_idx_) {
      return range;
    }
  }
  return std::nullopt;
}

std::optional<cached_run_shape> make_cached_shape(
    n::rt::frun const& fr, n::shapes_storage const& shapes) {
  if (!fr.is_scheduled() || fr.size() < 2U) {
    return std::nullopt;
  }

  auto const trip = fr.trip_idx();
  auto const trip_range = get_trip_range(fr, trip);
  if (!trip_range.has_value() || fr.stop_range_.from_ < trip_range->from_ ||
      fr.stop_range_.to_ > trip_range->to_) {
    return std::nullopt;
  }

  if (trip >= shapes.trip_offset_indices_.size()) {
    return std::nullopt;
  }
  auto const [shape_idx, offset_idx] = shapes.trip_offset_indices_[trip];
  auto const shape = shapes.get_shape(shape_idx);
  if (shape.size() < 2U || offset_idx == n::shape_offset_idx_t::invalid() ||
      offset_idx >= shapes.offsets_.size()) {
    return std::nullopt;
  }

  auto const offsets = shapes.offsets_[offset_idx];
  auto const local_from =
      static_cast<n::stop_idx_t>(fr.stop_range_.from_ - trip_range->from_);
  auto const local_to =
      static_cast<n::stop_idx_t>(fr.stop_range_.to_ - trip_range->from_);
  if (local_to > offsets.size() || local_to <= local_from + 1U) {
    return std::nullopt;
  }

  auto const first_point = static_cast<unsigned>(offsets[local_from]);
  auto const last_point =
      static_cast<unsigned>(offsets[static_cast<n::stop_idx_t>(local_to - 1U)]);
  if (first_point >= shape.size() || last_point >= shape.size() ||
      first_point >= last_point) {
    return std::nullopt;
  }

  auto cached = cached_run_shape{};
  cached.points_.assign(std::next(begin(shape), first_point),
                        std::next(begin(shape), last_point + 1U));
  cached.point_distances_.reserve(cached.points_.size());
  cached.point_distances_.push_back(0.0);
  for (auto i = std::size_t{1U}; i != cached.points_.size(); ++i) {
    cached.point_distances_.push_back(
        cached.point_distances_.back() +
        geo::distance(cached.points_[i - 1U], cached.points_[i]));
  }

  for (auto stop = local_from; stop != local_to; ++stop) {
    auto const point = static_cast<unsigned>(offsets[stop]) - first_point;
    if (point >= cached.point_distances_.size()) {
      return std::nullopt;
    }
    cached.stop_point_indices_.push_back(point);
    cached.stop_distances_.push_back(cached.point_distances_[point]);
  }

  auto const seq_range = n::loader::gtfs::stop_seq_number_range{
      {fr.tt_->trip_stop_seq_numbers_[trip]},
      static_cast<n::stop_idx_t>(trip_range->size())};
  auto all_sequences = std::vector<unsigned>{};
  for (auto const sequence : seq_range) {
    all_sequences.push_back(sequence);
  }
  if (local_to > all_sequences.size()) {
    return std::nullopt;
  }
  cached.static_stop_sequences_.assign(
      std::next(begin(all_sequences), local_from),
      std::next(begin(all_sequences), local_to));
  return cached;
}

}  // namespace

struct trip_progress_projector::impl {
  std::map<cache_key, cached_run_shape> cache_;
};

trip_progress_projector::trip_progress_projector()
    : impl_{std::make_unique<impl>()} {}

trip_progress_projector::~trip_progress_projector() = default;

trip_progress_projector::trip_progress_projector(
    trip_progress_projector&&) noexcept = default;

trip_progress_projector& trip_progress_projector::operator=(
    trip_progress_projector&&) noexcept = default;

trip_progress_projection trip_progress_projector::project(
    n::rt::frun const& fr,
    n::shapes_storage const& shapes,
    geo::latlng const& position,
    std::optional<trip_progress> const& prior) {
  if (!fr.is_scheduled()) {
    return {};
  }
  auto const key = cache_key{&shapes, fr.trip_idx(), fr.stop_range_.from_,
                             fr.stop_range_.to_};
  auto it = impl_->cache_.find(key);
  if (it == end(impl_->cache_)) {
    auto shape = make_cached_shape(fr, shapes);
    if (!shape.has_value()) {
      return {};
    }
    it = impl_->cache_.emplace(key, std::move(*shape)).first;
  }

  struct projection_candidate {
    double lateral_error_;
    double distance_along_;
    std::size_t next_stop_idx_;
  };

  auto const& shape = it->second;
  auto candidates = std::vector<projection_candidate>{};
  for (auto section = std::size_t{0U};
       section + 1U < shape.stop_point_indices_.size(); ++section) {
    auto const from = shape.stop_point_indices_[section];
    auto const to = shape.stop_point_indices_[section + 1U];
    if (from >= to) {
      continue;
    }
    auto const section_shape =
        std::span{shape.points_}.subspan(from, to - from + 1U);
    auto const projected = geo::distance_to_polyline(position, section_shape);
    auto const segment = from + projected.segment_idx_;
    candidates.push_back(
        {.lateral_error_ = projected.distance_to_polyline_,
         .distance_along_ =
             shape.point_distances_[segment] +
             geo::distance(shape.points_[segment], projected.best_),
         .next_stop_idx_ = section + 1U});
  }
  if (candidates.empty()) {
    return {};
  }

  auto const best_lateral =
      std::ranges::min(candidates, {}, &projection_candidate::lateral_error_)
          .lateral_error_;
  constexpr auto kMaxLateralErrorMeters = 100.0;
  if (best_lateral > kMaxLateralErrorMeters) {
    return {.status_ = trip_progress_projection_status::kOffShape};
  }

  constexpr auto kEquivalentLateralErrorMeters = 5.0;
  std::erase_if(candidates, [&](projection_candidate const& candidate) {
    return candidate.lateral_error_ >
           best_lateral + kEquivalentLateralErrorMeters;
  });
  std::ranges::sort(candidates, {}, &projection_candidate::distance_along_);
  constexpr auto kEquivalentProgressMeters = 25.0;
  if (!prior.has_value() && candidates.size() > 1U &&
      candidates.back().distance_along_ - candidates.front().distance_along_ >
          kEquivalentProgressMeters) {
    return {.status_ = trip_progress_projection_status::kAmbiguous};
  }

  if (prior.has_value()) {
    std::erase_if(candidates, [&](projection_candidate const& candidate) {
      return candidate.distance_along_ + kEquivalentProgressMeters <
             prior->distance_along_shape_m_;
    });
    if (candidates.empty()) {
      return {.status_ = trip_progress_projection_status::kImplausible};
    }
    std::ranges::sort(candidates, [&](projection_candidate const& a,
                                      projection_candidate const& b) {
      return std::tuple{
                 a.lateral_error_,
                 std::abs(a.distance_along_ - prior->distance_along_shape_m_)} <
             std::tuple{
                 b.lateral_error_,
                 std::abs(b.distance_along_ - prior->distance_along_shape_m_)};
    });
  }

  auto const& candidate = candidates.front();
  auto monotonicity = trip_progress_monotonicity::kNoPrior;
  if (prior.has_value()) {
    auto const delta =
        candidate.distance_along_ - prior->distance_along_shape_m_;
    monotonicity = delta > 1.0    ? trip_progress_monotonicity::kForward
                   : delta < -1.0 ? trip_progress_monotonicity::kMinorRegression
                                  : trip_progress_monotonicity::kStationary;
  }
  return {.status_ = trip_progress_projection_status::kProjected,
          .progress_ = trip_progress{
              .distance_along_shape_m_ = candidate.distance_along_,
              .lateral_error_m_ = candidate.lateral_error_,
              .next_static_stop_sequence_ =
                  shape.static_stop_sequences_[candidate.next_stop_idx_],
              .distance_to_next_stop_m_ = std::max(
                  0.0, shape.stop_distances_[candidate.next_stop_idx_] -
                           candidate.distance_along_),
              .monotonicity_ = monotonicity}};
}

}  // namespace motis
