#include "motis/update_rtt_td_footpaths.h"

#include <map>

#include "utl/equal_ranges_linear.h"
#include "utl/parallel_for.h"

#include "osr/routing/parameters.h"
#include "osr/routing/route.h"

#include "motis/constants.h"
#include "motis/get_loc.h"
#include "motis/get_stops_with_traffic.h"
#include "motis/osr/max_distance.h"

namespace n = nigiri;
using namespace std::chrono_literals;

namespace motis {

using node_states_t =
    std::pair<nodes_t, std::vector<std::pair<n::unixtime_t, states_t>>>;

node_states_t get_node_states(osr::ways const& w,
                              osr::lookup const& l,
                              elevators const& e,
                              geo::latlng const& pos) {
  auto e_nodes =
      utl::to_vec(l.find_elevators(geo::box{pos, kElevatorUpdateRadius}));
  auto e_state_changes =
      get_state_changes(
          utl::to_vec(
              e_nodes,
              [&](osr::node_idx_t const n)
                  -> std::vector<state_change<n::unixtime_t>> {
                auto const ne =
                    match_elevator(e.elevators_rtree_, e.elevators_, w, n);
                if (ne == elevator_idx_t::invalid()) {
                  return {
                      {.valid_from_ = n::unixtime_t{n::unixtime_t::duration{0}},
                       .state_ = true}};
                }
                return e.elevators_[ne].get_state_changes();
              }))
          .to_vec();
  return {std::move(e_nodes), std::move(e_state_changes)};
}

osr::bitvec<osr::node_idx_t>& set_blocked(
    nodes_t const& e_nodes,
    states_t const& states,
    osr::bitvec<osr::node_idx_t>& blocked_mem) {
  blocked_mem.zero_out();
  for (auto const [n, s] : utl::zip(e_nodes, states)) {
    blocked_mem.set(n, !s);
  }
  return blocked_mem;
}

std::optional<std::pair<nodes_t, states_t>> get_states_at(
    osr::ways const& w,
    osr::lookup const& l,
    elevators const& e,
    n::unixtime_t const t,
    geo::latlng const& pos) {
  auto const [e_nodes, e_state_changes] = get_node_states(w, l, e, pos);
  if (e_nodes.empty()) {
    return std::pair{nodes_t{}, states_t{}};
  }
  auto const it = std::lower_bound(
      begin(e_state_changes), end(e_state_changes), t,
      [&](auto&& a, n::unixtime_t const b) { return a.first < b; });
  if (it == begin(e_state_changes)) {
    return std::nullopt;
  }
  return std::pair{e_nodes, std::prev(it)->second};
}

std::vector<n::td_footpath> get_td_footpaths(
    osr::ways const& w,
    osr::lookup const& l,
    osr::platforms const& pl,
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    point_rtree<n::location_idx_t> const& loc_rtree,
    elevators const& e,
    platform_matches_t const& matches,
    n::location_idx_t const start_l,
    osr::location const start,
    osr::direction const dir,
    osr::search_profile const profile,
    std::chrono::seconds const max,
    double const max_matching_distance,
    osr_parameters const& osr_params,
    osr::bitvec<osr::node_idx_t>& blocked_mem) {
  blocked_mem.resize(w.n_nodes());

  auto const [e_nodes, e_state_changes] = get_node_states(w, l, e, start.pos_);

  auto fps = std::vector<n::td_footpath>{};
  for (auto const& [t, states] : e_state_changes) {
    set_blocked(e_nodes, states, blocked_mem);

    auto const neighbors = get_stops_with_traffic(
        tt, rtt, loc_rtree, start, get_max_distance(profile, max), start_l);
    auto const results = osr::route(
        to_profile_parameters(profile, osr_params), w, l, profile, start,
        utl::to_vec(neighbors,
                    [&](auto&& x) { return get_loc(tt, w, pl, matches, x); }),
        static_cast<osr::cost_t>(max.count()), dir, max_matching_distance,
        &blocked_mem);

    for (auto const [to, p] : utl::zip(neighbors, results)) {
      auto const duration = p.has_value() && (n::duration_t{p->cost_ / 60U} <
                                              n::footpath::kMaxDuration)
                                ? n::duration_t{p->cost_ / 60U}
                                : n::footpath::kMaxDuration;
      fps.push_back(n::td_footpath{
          to, t,
          n::duration_t{std::max(n::duration_t::rep{1}, duration.count())}});
    }
  }

  utl::sort(fps);

  utl::equal_ranges_linear(
      fps, [](auto const& a, auto const& b) { return a.target_ == b.target_; },
      [&](std::vector<n::td_footpath>::iterator& lb,
          std::vector<n::td_footpath>::iterator& ub) {
        for (auto it = lb; it != ub; ++it) {
          if (it->duration_ == n::footpath::kMaxDuration && it != lb &&
              (it - 1)->duration_ != n::footpath::kMaxDuration) {
            // TODO support feasible, but longer paths
            it->valid_from_ -= (it - 1)->duration_ - n::duration_t{1U};
          }
        }
      });

  return fps;
}

void update_rtt_td_footpaths(
    osr::ways const& w,
    osr::lookup const& l,
    osr::platforms const& pl,
    nigiri::timetable const& tt,
    point_rtree<n::location_idx_t> const& loc_rtree,
    elevators const& e,
    platform_matches_t const& matches,
    hash_set<std::pair<n::location_idx_t, osr::direction>> const& tasks,
    nigiri::rt_timetable const* old_rtt,
    nigiri::rt_timetable& rtt,
    std::chrono::seconds const max) {
  auto in_mutex = std::mutex{}, out_mutex = std::mutex{};
  auto out = std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
  auto in = std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
  utl::parallel_for_run_threadlocal<osr::bitvec<osr::node_idx_t>>(
      tasks.size(),
      [&](osr::bitvec<osr::node_idx_t>& blocked, std::size_t const task_idx) {
        auto const [start, dir] = *(begin(tasks) + task_idx);
        auto fps = get_td_footpaths(w, l, pl, tt, &rtt, loc_rtree, e, matches,
                                    start, get_loc(tt, w, pl, matches, start),
                                    dir, osr::search_profile::kWheelchair, max,
                                    kMaxWheelchairMatchingDistance,
                                    osr_parameters{}, blocked);
        {
          auto const lock = std::unique_lock{
              dir == osr::direction::kForward ? out_mutex : in_mutex};
          (dir == osr::direction::kForward ? out : in)[start] = std::move(fps);
        }
      });

  rtt.td_footpaths_out_[2].clear();
  for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
    auto const it = out.find(i);
    if (it != end(out)) {
      rtt.has_td_footpaths_out_[2].set(i, true);
      rtt.td_footpaths_out_[2].emplace_back(it->second);
    } else if (old_rtt != nullptr) {
      rtt.has_td_footpaths_out_[2].set(
          i, old_rtt->has_td_footpaths_out_[2].test(i));
      rtt.td_footpaths_out_[2].emplace_back(old_rtt->td_footpaths_out_[2][i]);
    } else {
      rtt.has_td_footpaths_out_[2].set(i, false);
      rtt.td_footpaths_out_[2].emplace_back(
          std::initializer_list<n::td_footpath>{});
    }
  }

  rtt.td_footpaths_in_[2].clear();
  for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
    auto const it = in.find(i);
    if (it != end(in)) {
      rtt.has_td_footpaths_in_[2].set(i, true);
      rtt.td_footpaths_in_[2].emplace_back(it->second);
    } else if (old_rtt != nullptr) {
      rtt.has_td_footpaths_in_[2].set(i,
                                      old_rtt->has_td_footpaths_in_[2].test(i));
      rtt.td_footpaths_in_[2].emplace_back(old_rtt->td_footpaths_in_[2][i]);
    } else {
      rtt.has_td_footpaths_in_[2].set(i, false);
      rtt.td_footpaths_in_[2].emplace_back(
          std::initializer_list<n::td_footpath>{});
    }
  }
}

void update_rtt_td_footpaths(osr::ways const& w,
                             osr::lookup const& l,
                             osr::platforms const& pl,
                             nigiri::timetable const& tt,
                             point_rtree<n::location_idx_t> const& loc_rtree,
                             elevators const& e,
                             elevator_footpath_map_t const& elevators_in_paths,
                             platform_matches_t const& matches,
                             nigiri::rt_timetable& rtt,
                             std::chrono::seconds const max) {
  auto tasks = hash_set<std::pair<n::location_idx_t, osr::direction>>{};
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
  update_rtt_td_footpaths(w, l, pl, tt, loc_rtree, e, matches, tasks, nullptr,
                          rtt, max);
}

}  // namespace motis
