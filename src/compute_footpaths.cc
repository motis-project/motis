#include "motis/compute_footpaths.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/parallel_for.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"
#include "osr/util/infinite.h"
#include "osr/util/reverse.h"

#include "motis/constants.h"
#include "motis/get_loc.h"
#include "motis/match_platforms.h"
#include "motis/point_rtree.h"

namespace n = nigiri;

namespace motis {

vector_map<n::location_idx_t, osr::match_t> lookup_locations(
    osr::ways const& w,
    osr::lookup const& lookup,
    osr::platforms const& pl,
    n::timetable const& tt,
    platform_matches_t const& matches,
    osr::search_profile const profile) {
  auto const timer = utl::scoped_timer{fmt::format(
      "matching timetable locations for profile={}", to_str(profile))};

  auto ret = vector_map<n::location_idx_t, osr::match_t>{};
  ret.resize(tt.n_locations());

  utl::parallel_for_run(tt.n_locations(), [&](std::size_t const x) {
    auto const l =
        n::location_idx_t{static_cast<n::location_idx_t::value_t>(x)};
    // - fixed `direction=forward` only works because we don't reconstruct and
    //   because foot/wheelchair can use ways
    // - fixed `reverse=false` only works because foot/wheelchair can use ways
    //   in both directions.
    ret[l] = lookup.match(get_loc(tt, w, pl, matches, l), false,
                          osr::direction::kForward, kMaxMatchingDistance,
                          nullptr, profile);
  });

  return ret;
}

elevator_footpath_map_t compute_footpaths(osr::ways const& w,
                                          osr::lookup const& lookup,
                                          osr::platforms const& pl,
                                          nigiri::timetable& tt,
                                          bool const update_coordinates) {
  fmt::println(std::clog, "  -> creating matches");
  auto const matches = get_matches(tt, pl, w);

  fmt::println(std::clog, "  -> creating r-tree");
  auto const loc_rtree = [&]() {
    auto t = point_rtree<n::location_idx_t>{};
    for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
      if (update_coordinates && matches[i] != osr::platform_idx_t::invalid()) {
        auto const center = get_platform_center(pl, w, matches[i]);
        if (center.has_value() &&
            geo::distance(*center, tt.locations_.coordinates_[i]) <
                kMaxAdjust) {
          tt.locations_.coordinates_[i] = *center;
        }
      }

      if (!tt.location_routes_[i].empty()) {
        t.add(tt.locations_.coordinates_[i], i);
      }
    }
    return t;
  }();

  auto const pt = utl::get_active_progress_tracker();
  pt->in_high(tt.n_locations() * 2U);

  auto footpaths_out_foot =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_out_foot.resize(tt.n_locations());
  auto footpaths_out_wheelchair =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_out_wheelchair.resize(tt.n_locations());

  auto elevator_in_paths_mutex = std::mutex{};
  auto elevator_in_paths = elevator_footpath_map_t{};
  auto const add_if_elevator = [&](osr::node_idx_t const n,
                                   n::location_idx_t const a,
                                   n::location_idx_t const b) {
    if (n != osr::node_idx_t::invalid() &&
        w.r_->node_properties_[n].is_elevator()) {
      auto l = std::unique_lock{elevator_in_paths_mutex};
      elevator_in_paths[n].emplace(a, b);
    }
  };

  auto const foot_candidates =
      lookup_locations(w, lookup, pl, tt, matches, osr::search_profile::kFoot);
  auto const wheelchair_candidates = lookup_locations(
      w, lookup, pl, tt, matches, osr::search_profile::kWheelchair);

  auto m = std::mutex{};
  for (auto const mode :
       {osr::search_profile::kFoot, osr::search_profile::kWheelchair}) {
    auto const& candidates = mode == osr::search_profile::kFoot
                                 ? foot_candidates
                                 : wheelchair_candidates;
    utl::parallel_for_run(tt.n_locations(), [&](auto const i) {
      auto const l = n::location_idx_t{i};
      auto& footpaths =
          (mode == osr::search_profile::kFoot ? footpaths_out_foot[l]
                                              : footpaths_out_wheelchair[l]);
      auto neighbors = std::vector<n::location_idx_t>{};
      loc_rtree.in_radius(tt.locations_.coordinates_[l], kMaxDistance,
                          [&](n::location_idx_t const x) {
                            if (x != l) {
                              neighbors.emplace_back(x);
                            }
                          });
      auto const results = osr::route(
          w, mode, get_loc(tt, w, pl, matches, l),
          utl::to_vec(neighbors,
                      [&](auto&& x) { return get_loc(tt, w, pl, matches, x); }),
          candidates[l],
          utl::to_vec(neighbors, [&](auto&& x) { return candidates[x]; }),
          kMaxDuration, osr::direction::kForward, nullptr,
          [](osr::path const& p) { return p.uses_elevator_; });
      for (auto const [n, r] : utl::zip(neighbors, results)) {
        if (r.has_value()) {
          auto lock = std::scoped_lock{m};
          auto const duration = n::duration_t{r->cost_ / 60U};
          if (duration < n::footpath::kMaxDuration) {
            footpaths.emplace_back(n::footpath{n, duration});
          }
          for (auto const& s : r->segments_) {
            add_if_elevator(s.from_, l, n);
            add_if_elevator(s.from_, n, l);
          }
        }
      }

      utl::sort(footpaths_out_foot[l]);
      utl::sort(footpaths_out_wheelchair[l]);

      pt->update_monotonic(
          (mode == osr::search_profile::kFoot ? 0U : tt.n_locations()) + i);
    });
  }

  fmt::println(std::clog, "  -> create ingoing footpaths");
  auto footpaths_in_foot =
      n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
  footpaths_in_foot.resize(tt.n_locations());
  for (auto const [i, out] : utl::enumerate(footpaths_out_foot)) {
    auto const l = n::location_idx_t{i};
    for (auto const fp : out) {
      assert(fp.target() < tt.n_locations());
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
      assert(fp.target() < tt.n_locations());
      footpaths_in_wheelchair[fp.target()].emplace_back(
          n::footpath{l, fp.duration()});
    }
  }

  fmt::println(std::clog, "  -> copy footpaths");
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

  return elevator_in_paths;
}

}  // namespace motis