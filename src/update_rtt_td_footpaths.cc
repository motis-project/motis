#include "icc/update_rtt_td_footpaths.h"

#include <map>

#include "utl/parallel_for.h"

#include "osr/routing/route.h"

#include "icc/constants.h"
#include "icc/get_loc.h"

namespace n = nigiri;
using namespace std::chrono_literals;

namespace icc {

std::vector<n::td_footpath> get_td_footpaths(
    osr::ways const& w,
    osr::lookup const& l,
    osr::platforms const& pl,
    nigiri::timetable const& tt,
    point_rtree<n::location_idx_t> const& loc_rtree,
    elevators const& e,
    elevator_footpath_map_t const& elevators_in_paths,
    platform_matches_t const& matches,
    osr::location const start,
    osr::direction const dir,
    osr::search_profile const profile,
    osr::bitvec<osr::node_idx_t>& blocked) {
  blocked.resize(w.n_nodes());

  auto const e_nodes = l.find_elevators(geo::box{start.pos_, 1000});
  auto const e_elevators = utl::to_vec(e_nodes, [&](auto&& x) {
    return match_elevator(e.elevators_rtree_, e.elevators_, w, x);
  });
  auto const e_state_changes =
      get_state_changes(
          utl::to_vec(
              e_elevators,
              [&](elevator_idx_t const ne)
                  -> std::vector<state_change<n::unixtime_t>> {
                if (ne == elevator_idx_t::invalid()) {
                  return {
                      {.valid_from_ = n::unixtime_t{n::unixtime_t::duration{0}},
                       .state_ = true}};
                }
                return e.elevators_[ne].get_state_changes();
              }))
          .to_vec();

  auto fps = std::vector<n::td_footpath>{};
  for (auto const& [t, states] : e_state_changes) {
    blocked.zero_out();
    for (auto const [n, s] : utl::zip(e_nodes, states)) {
      blocked.set(n, !s);
    }

    auto neighbors = std::vector<n::location_idx_t>{};
    loc_rtree.in_radius(
        start.pos_, kMaxDistance,
        [&](n::location_idx_t const x) { neighbors.emplace_back(x); });
    auto const results = osr::route(
        w, l, profile, start,
        utl::to_vec(neighbors,
                    [&](auto&& x) { return get_loc(tt, w, pl, matches, x); }),
        kMaxDuration, dir, kMaxMatchingDistance, &blocked);

    for (auto const [to, p] : utl::zip(neighbors, results)) {
      auto const duration = p.has_value() && (n::duration_t{p->cost_ / 60U} <
                                              n::footpath::kMaxDuration)
                                ? n::duration_t{p->cost_ / 60U}
                                : n::footpath::kMaxDuration;
      fps.emplace_back(
          to, t,
          n::duration_t{std::max(n::duration_t::rep{1}, duration.count())});
    }
  }

  utl::sort(fps);

  return fps;
}

void update_rtt_td_footpaths(osr::ways const& w,
                             osr::lookup const& l,
                             osr::platforms const& pl,
                             nigiri::timetable const& tt,
                             point_rtree<n::location_idx_t> const& loc_rtree,
                             elevators const& e,
                             elevator_footpath_map_t const& elevators_in_paths,
                             platform_matches_t const& matches,
                             nigiri::rt_timetable& rtt) {
  auto tasks = hash_set<std::pair<n::location_idx_t, osr::direction>>{};
  auto n_sources = 0U, n_sinks = 0U;
  for (auto const& [e_in_path, from_to] : elevators_in_paths) {
    auto const e_idx =
        match_elevator(e.elevators_rtree_, e.elevators_, w, e_in_path);
    if (e_idx == elevator_idx_t::invalid()) {
      continue;
    }
    auto const& el = e.elevators_[e_idx];
    if (el.out_of_service_.empty() && el.status_) {
      continue;
    }
    for (auto const& [from, to] : from_to) {
      tasks.emplace(from, osr::direction::kForward);
      tasks.emplace(to, osr::direction::kBackward);
    }
  }
  fmt::println("  -> {} routing tasks tasks", tasks.size());

  auto in_mutex = std::mutex{}, out_mutex = std::mutex{};
  auto out = std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
  auto in = std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
  utl::parallel_for_run_threadlocal<osr::bitvec<osr::node_idx_t>>(
      tasks.size(),
      [&](osr::bitvec<osr::node_idx_t>& blocked, std::size_t const task_idx) {
        auto const [start, dir] = *(begin(tasks) + task_idx);
        auto fps =
            get_td_footpaths(w, l, pl, tt, loc_rtree, e, elevators_in_paths,
                             matches, get_loc(tt, w, pl, matches, start), dir,
                             osr::search_profile::kWheelchair, blocked);
        {
          auto const lock = std::unique_lock{
              dir == osr::direction::kForward ? out_mutex : in_mutex};
          (dir == osr::direction::kForward ? out : in)[start] = std::move(fps);
        }
      });

  for (auto& [from, footpaths] : out) {
    for (auto const& fp : footpaths) {
      rtt.has_td_footpaths_[2].set(from, true);
      rtt.td_footpaths_out_[2][from].push_back(fp);
    }
  }

  for (auto& [from, footpaths] : in) {
    for (auto const& fp : footpaths) {
      rtt.has_td_footpaths_[2].set(from, true);
      rtt.td_footpaths_in_[2][from].push_back(fp);
    }
  }
}

}  // namespace icc