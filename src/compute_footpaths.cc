#include "icc/compute_footpaths.h"

#include "utl/parallel_for.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

#include "icc/match_platforms.h"
#include "icc/point_rtree.h"

namespace n = nigiri;

namespace icc {

constexpr auto const kMaxDuration = 15;
constexpr auto const kMaxDistance = 2000;

void compute_footpaths(nigiri::timetable& tt,
                       osr::ways const& w,
                       osr::lookup const& lookup,
                       osr::platforms const& pl,
                       osr::bitvec<osr::node_idx_t> const& blocked,
                       bool const update_coordinates) {
  fmt::println("creating matches");
  auto const matches = [&]() {
    auto m = n::vector_map<n::location_idx_t, osr::platform_idx_t>{};
    m.resize(tt.n_locations());
    utl::parallel_for_run(tt.n_locations(), [&](auto const i) {
      auto const l = n::location_idx_t{i};
      m[l] = get_match(tt, pl, w, l);
    });
    return m;
  }();

  auto const get_loc = [&](n::location_idx_t const l) -> osr::location {
    return {tt.locations_.coordinates_[l],
            matches[l] == osr::platform_idx_t::invalid()
                ? osr::to_level(0.0F)
                : pl.get_level(w, matches[l])};
  };

  fmt::println("creating r-tree");
  auto const loc_rtree = [&]() {
    auto t = icc::point_rtree<n::location_idx_t>{};
    for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
      if (update_coordinates && matches[i] != osr::platform_idx_t::invalid()) {
        auto const center = get_platform_center(pl, w, matches[i]);
        if (center.has_value() &&
            geo::distance(*center, tt.locations_.coordinates_[i]) < 35) {
          tt.locations_.coordinates_[i] = *center;
        }
      }

      if (!tt.location_routes_[i].empty()) {
        t.add(tt.locations_.coordinates_[i], i);
      }
    }
    return t;
  }();

  auto const in_radius = [&](n::location_idx_t const l) {
    auto const l_pos = tt.locations_.coordinates_[l];
    auto v = std::vector<n::location_idx_t>{};
    loc_rtree.find(l_pos, [&](n::location_idx_t const x) {
      if (geo::distance(l_pos, tt.locations_.coordinates_[x]) < kMaxDistance) {
        v.emplace_back(x);
      }
    });
    return v;
  };

  auto const g = utl::global_progress_bars{};
  auto const pt = utl::get_active_progress_tracker_or_activate("routing");
  pt->in_high(tt.n_locations() * 2U);
  auto footpaths_out_foot =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_out_foot.resize(tt.n_locations());
  auto footpaths_out_wheelchair =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_out_wheelchair.resize(tt.n_locations());

  for (auto const mode :
       {osr::search_profile::kFoot, osr::search_profile::kWheelchair}) {
    utl::parallel_for_run(tt.n_locations(), [&](auto const i) {
      auto const l = n::location_idx_t{i};
      auto const neighbors = in_radius(l);
      auto const results = osr::route(
          w, lookup, mode, get_loc(l),
          utl::to_vec(neighbors, [&](auto&& l) { return get_loc(l); }),
          kMaxDuration, osr::direction::kForward, 8, &blocked);
      for (auto const [n, r] : utl::zip(neighbors, results)) {
        if (r.has_value()) {
          auto const duration = n::duration_t{r->cost_ / 60U};
          if (duration < n::footpath::kMaxDuration) {
            (mode == osr::search_profile::kFoot ? footpaths_out_foot[l]
                                                : footpaths_out_wheelchair[l])
                .emplace_back(n::footpath{n, duration});
          }
        }
      }

      pt->update_monotonic(
          (mode == osr::search_profile::kFoot ? 0U : tt.n_locations()) + i);
    });
  }

  fmt::println("create ingoing footpaths");
  auto footpaths_in_foot =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_in_foot.resize(tt.n_locations());
  for (auto const [i, out] : utl::enumerate(footpaths_out_foot)) {
    auto const l = n::location_idx_t{i};
    for (auto const fp : out) {
      footpaths_in_foot[fp.target()].emplace_back(
          n::footpath{l, fp.duration()});
    }
  }

  auto footpaths_in_wheelchair =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_in_wheelchair.resize(tt.n_locations());
  for (auto const [i, out] : utl::enumerate(footpaths_out_wheelchair)) {
    auto const l = n::location_idx_t{i};
    for (auto const fp : out) {
      footpaths_in_wheelchair[fp.target()].emplace_back(
          n::footpath{l, fp.duration()});
    }
  }

  fmt::println("copy footpaths");
  for (auto const& x : footpaths_out_foot) {
    tt.locations_.footpaths_out_[1].emplace_back(x);
  }
  for (auto const& x : footpaths_in_foot) {
    tt.locations_.footpaths_in_[1].emplace_back(x);
  }
  for (auto const& x : footpaths_out_wheelchair) {
    tt.locations_.footpaths_out_[2].emplace_back(x);
  }
  for (auto const& x : footpaths_in_wheelchair) {
    tt.locations_.footpaths_in_[2].emplace_back(x);
  }
}

}  // namespace icc