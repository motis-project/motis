#include "icc/update_rtt_td_footpaths.h"

#include <map>

#include "osr/routing/route.h"

#include "icc/constants.h"
#include "icc/get_loc.h"

namespace n = nigiri;

namespace icc {

void update_rtt_td_footpaths(osr::ways const& w,
                             osr::lookup const& l,
                             osr::platforms const& pl,
                             nigiri::timetable const& tt,
                             elevators const& e,
                             elevator_footpath_map_t const& elevators_in_paths,
                             platform_matches_t const& matches,
                             nigiri::rt_timetable& rtt) {
  auto td_footpaths_out =
      std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
  auto td_footpaths_in =
      std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
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

    auto const e_nodes = l.find_elevators(geo::box{el.pos_, 1000});
    auto const e_elevators = utl::to_vec(e_nodes, [&](auto&& x) {
      return match_elevator(e.elevators_rtree_, e.elevators_, w, x);
    });
    auto const e_state_changes =
        get_state_changes(
            utl::to_vec(e_elevators,
                        [&](elevator_idx_t const ne)
                            -> std::vector<state_change<n::unixtime_t>> {
                          if (ne == elevator_idx_t::invalid()) {
                            return {
                                {.valid_from_ =
                                     n::unixtime_t{n::unixtime_t::duration{0}},
                                 .state_ = true}};
                          }
                          return e.elevators_[ne].get_state_changes();
                        }))
            .to_vec();

    auto blocked = osr::bitvec<osr::node_idx_t>{w.n_nodes()};
    for (auto const& [t, states] : e_state_changes) {
      blocked.zero_out();
      for (auto const [n, s] : utl::zip(e_nodes, states)) {
        blocked.set(n, !s);
      }
      for (auto const& [from, to] : from_to) {
        auto const p = osr::route(w, l, osr::search_profile::kWheelchair,
                                  get_loc(tt, w, pl, matches, from),
                                  get_loc(tt, w, pl, matches, to), kMaxDuration,
                                  osr::direction::kForward,
                                  kMaxMatchingDistance, &blocked);
        if (p.has_value()) {
          td_footpaths_out[from].emplace_back(to, t,
                                              n::duration_t{p->cost_ / 60});
          td_footpaths_in[to].emplace_back(from, t,
                                           n::duration_t{p->cost_ / 60});
        }
      }
    }
  }

  for (auto& [from, footpaths] : td_footpaths_out) {
    utl::sort(footpaths);
    for (auto const& fp : footpaths) {
      rtt.has_td_footpaths_[2].set(from, true);
      rtt.td_footpaths_out_[2][from].push_back(fp);
    }
  }

  for (auto& [from, footpaths] : td_footpaths_in) {
    utl::sort(footpaths);
    for (auto const& fp : footpaths) {
      rtt.has_td_footpaths_[2].set(from, true);
      rtt.td_footpaths_in_[2][from].push_back(fp);
    }
  }
}

}  // namespace icc