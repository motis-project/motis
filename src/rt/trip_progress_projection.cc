#include "motis/rt/trip_progress_projection.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <list>
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

struct projection_candidate {
  double lateral_error_;
  double distance_along_;
  std::size_t next_stop_idx_;
};

struct candidate_selection {
  trip_progress_projection_status status_;
  std::optional<projection_candidate> candidate_;
};

struct cache_key {
  n::trip_idx_t trip_;
  n::stop_idx_t from_;
  n::stop_idx_t to_;

  bool operator<(cache_key const& other) const {
    return std::tuple{cista::to_idx(trip_), cista::to_idx(from_),
                      cista::to_idx(to_)} <
           std::tuple{cista::to_idx(other.trip_), cista::to_idx(other.from_),
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
  auto const [_shape_idx, offset_idx] = shapes.trip_offset_indices_[trip];
  if (offset_idx == n::shape_offset_idx_t::invalid() ||
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
  auto const full_shape = shapes.get_shape(trip);
  if (first_point >= full_shape.size() || last_point >= full_shape.size() ||
      first_point > last_point) {
    return std::nullopt;
  }

  auto const shape =
      shapes.get_shape(trip, n::interval<n::stop_idx_t>{local_from, local_to});
  if (shape.empty()) {
    return std::nullopt;
  }

  auto cached = cached_run_shape{};
  cached.points_.assign(begin(shape), end(shape));
  cached.point_distances_.reserve(cached.points_.size());
  cached.point_distances_.push_back(0.0);
  for (auto i = std::size_t{1U}; i != cached.points_.size(); ++i) {
    cached.point_distances_.push_back(
        cached.point_distances_.back() +
        geo::distance(cached.points_[i - 1U], cached.points_[i]));
  }

  auto previous_point = first_point;
  for (auto stop = local_from; stop != local_to; ++stop) {
    auto const absolute_point = static_cast<unsigned>(offsets[stop]);
    if (absolute_point < previous_point) {
      return std::nullopt;
    }
    previous_point = absolute_point;
    auto const point = absolute_point - first_point;
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

std::optional<std::size_t> get_constrained_stop_idx(
    cached_run_shape const& shape,
    vehicle_position_progress_constraint const& constraint) {
  auto const seq = std::ranges::find(shape.static_stop_sequences_,
                                     constraint.current_static_stop_sequence_);
  if (seq == end(shape.static_stop_sequences_)) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(
      std::distance(begin(shape.static_stop_sequences_), seq));
}

std::vector<projection_candidate> make_projection_candidates(
    cached_run_shape const& shape,
    geo::latlng const& position,
    std::optional<std::size_t> const constrained_stop_idx,
    bool const stopped_at) {
  auto candidates = std::vector<projection_candidate>{};
  if (stopped_at) {
    auto const stop = *constrained_stop_idx;
    auto const point = shape.stop_point_indices_[stop];
    candidates.push_back(
        {.lateral_error_ = geo::distance(position, shape.points_[point]),
         .distance_along_ = shape.stop_distances_[stop],
         .next_stop_idx_ = stop});
    return candidates;
  }

  for (auto section = std::size_t{0U};
       section + 1U < shape.stop_point_indices_.size(); ++section) {
    auto const next_stop = section + 1U;
    if (constrained_stop_idx.has_value() &&
        next_stop != *constrained_stop_idx) {
      continue;
    }
    auto const from = shape.stop_point_indices_[section];
    auto const to = shape.stop_point_indices_[next_stop];
    if (from > to) {
      continue;
    }
    if (from == to) {
      candidates.push_back(
          {.lateral_error_ = geo::distance(position, shape.points_[from]),
           .distance_along_ = shape.stop_distances_[next_stop],
           .next_stop_idx_ = next_stop});
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
         .next_stop_idx_ = next_stop});
  }
  return candidates;
}

candidate_selection select_candidate(
    std::vector<projection_candidate> candidates,
    std::optional<trip_progress> const& prior) {
  constexpr auto kEquivalentProgressMeters = 25.0;
  if (prior.has_value()) {
    std::erase_if(candidates, [&](projection_candidate const& candidate) {
      return candidate.distance_along_ + kEquivalentProgressMeters <
             prior->distance_along_shape_m_;
    });
    if (candidates.empty()) {
      return {.status_ = trip_progress_projection_status::kImplausible};
    }
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
  if (candidates.size() > 1U &&
      candidates.back().distance_along_ - candidates.front().distance_along_ >
          kEquivalentProgressMeters) {
    return {.status_ = trip_progress_projection_status::kAmbiguous};
  }
  constexpr auto kEndpointTieToleranceMeters = 1e-3;
  for (auto i = std::size_t{0U}; i != candidates.size(); ++i) {
    for (auto j = i + 1U; j != candidates.size(); ++j) {
      if (candidates[i].next_stop_idx_ != candidates[j].next_stop_idx_ &&
          std::abs(candidates[i].distance_along_ -
                   candidates[j].distance_along_) <=
              kEndpointTieToleranceMeters &&
          std::abs(candidates[i].lateral_error_ -
                   candidates[j].lateral_error_) <=
              kEndpointTieToleranceMeters) {
        return {.status_ = trip_progress_projection_status::kAmbiguous};
      }
    }
  }
  std::ranges::sort(candidates, {}, &projection_candidate::lateral_error_);
  return {.status_ = trip_progress_projection_status::kProjected,
          .candidate_ = candidates.front()};
}

}  // namespace

struct trip_progress_projector::impl {
  static constexpr auto kMaxCachedRunShapes = 1024U;

  struct cache_entry {
    cached_run_shape shape_;
    std::list<cache_key>::iterator lru_it_;
  };

  explicit impl(n::shapes_storage const& shapes) : shapes_{shapes} {}

  cached_run_shape const* get_shape(n::rt::frun const& fr) {
    auto const key =
        cache_key{fr.trip_idx(), fr.stop_range_.from_, fr.stop_range_.to_};
    if (auto const it = cache_.find(key); it != end(cache_)) {
      lru_.splice(end(lru_), lru_, it->second.lru_it_);
      return &it->second.shape_;
    }

    auto shape = make_cached_shape(fr, shapes_);
    if (!shape.has_value()) {
      return nullptr;
    }
    if (cache_.size() == kMaxCachedRunShapes) {
      cache_.erase(lru_.front());
      lru_.pop_front();
    }
    lru_.push_back(key);
    auto const it =
        cache_
            .emplace(key, cache_entry{std::move(*shape), std::prev(end(lru_))})
            .first;
    return &it->second.shape_;
  }

  n::shapes_storage const& shapes_;
  std::list<cache_key> lru_;
  std::map<cache_key, cache_entry> cache_;
};

trip_progress_projector::trip_progress_projector(
    n::shapes_storage const& shapes)
    : impl_{std::make_unique<impl>(shapes)} {}

trip_progress_projector::~trip_progress_projector() = default;

trip_progress_projector::trip_progress_projector(
    trip_progress_projector&&) noexcept = default;

trip_progress_projector& trip_progress_projector::operator=(
    trip_progress_projector&&) noexcept = default;

trip_progress_projection trip_progress_projector::project(
    n::rt::frun const& fr,
    geo::latlng const& position,
    std::optional<trip_progress> const& prior,
    std::optional<vehicle_position_progress_constraint> const& vp_constraint) {
  if (!fr.is_scheduled()) {
    return {};
  }
  auto const* shape = impl_->get_shape(fr);
  if (shape == nullptr) {
    return {};
  }

  auto constrained_stop_idx = std::optional<std::size_t>{};
  if (vp_constraint.has_value()) {
    constrained_stop_idx = get_constrained_stop_idx(*shape, *vp_constraint);
    if (!constrained_stop_idx.has_value()) {
      return {.status_ = trip_progress_projection_status::kImplausible};
    }
  }

  auto candidates = make_projection_candidates(
      *shape, position, constrained_stop_idx,
      vp_constraint.has_value() &&
          vp_constraint->status_ == vehicle_position_stop_status::kStoppedAt);
  if (candidates.empty()) {
    return {.status_ = vp_constraint.has_value()
                           ? trip_progress_projection_status::kImplausible
                           : trip_progress_projection_status::kMissingShape};
  }

  auto const selection = select_candidate(std::move(candidates), prior);
  if (!selection.candidate_.has_value()) {
    return {.status_ = selection.status_};
  }
  if (!vp_constraint.has_value() &&
      shape->stop_point_indices_.front() == shape->stop_point_indices_.back()) {
    return {.status_ = trip_progress_projection_status::kAmbiguous};
  }
  auto const& candidate = *selection.candidate_;
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
                  shape->static_stop_sequences_[candidate.next_stop_idx_],
              .distance_to_next_stop_m_ = std::max(
                  0.0, shape->stop_distances_[candidate.next_stop_idx_] -
                           candidate.distance_along_),
              .monotonicity_ = monotonicity}};
}

}  // namespace motis
