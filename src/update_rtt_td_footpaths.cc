#include "motis/update_rtt_td_footpaths.h"

#include <cassert>
#include <map>

#include "utl/equal_ranges_linear.h"
#include "utl/parallel_for.h"
#include "utl/raii.h"

#include "osr/routing/parameters.h"
#include "osr/routing/route.h"

#include "nigiri/td_footpath.h"

#include "motis/constants.h"
#include "motis/get_loc.h"
#include "motis/get_stops_with_traffic.h"
#include "motis/osr/footpath_restriction.h"
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
  for (auto const [n, s] : utl::zip(e_nodes, states)) {
    blocked_mem.set(n, !s);
  }
  return blocked_mem;
}

void reset_blocked(nodes_t const& e_nodes,
                   osr::bitvec<osr::node_idx_t>& blocked_mem) {
  for (auto const n : e_nodes) {
    blocked_mem.set(n, false);
  }
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
  // The last state change at or before `t`.
  auto const it = std::upper_bound(
      begin(e_state_changes), end(e_state_changes), t,
      [&](n::unixtime_t const a, auto&& b) { return a < b.first; });
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
  assert(blocked_mem.none());

  auto const [e_nodes, e_state_changes] = get_node_states(w, l, e, start.pos_);
  auto const reset =
      utl::finally{[&]() { reset_blocked(e_nodes, blocked_mem); }};

  auto fps = std::vector<n::td_footpath>{};
  for (auto const& [t, states] : e_state_changes) {
    set_blocked(e_nodes, states, blocked_mem);

    auto const neighbors = get_stops_with_traffic(
        tt, rtt, loc_rtree, start, get_max_distance(profile, osr_params, max),
        start_l);
    auto const results = osr::route(
        to_profile_parameters(profile, osr_params), w, l, profile, start,
        utl::to_vec(neighbors,
                    [&](auto&& x) { return get_loc(tt, w, pl, matches, x); }),
        max, dir, max_matching_distance, &blocked_mem);

    for (auto const [to, p] : utl::zip(neighbors, results)) {
      auto const duration =
          p.has_value()
              ? std::min(n::duration_t{static_cast<n::duration_t::rep>(
                             std::ceil(p->duration_.count() / 60.0))},
                         n::footpath::kMax)
              : n::kTdFootpathBan;
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
          if (it->duration_ == n::kTdFootpathBan && it != lb &&
              (it - 1)->duration_ != n::kTdFootpathBan) {
            // TODO support feasible, but longer paths
            it->valid_from_ -= (it - 1)->duration_ - n::duration_t{1U};
          }
        }
      });

  return fps;
}

// Time-dependent footpaths of the foot profile from start_l: one series per
// (target, restriction record) over the elevator state changes nearby (Pareto
// search with the out-of-service elevators blocked, ban while unreachable).
// Station graph pairs and the other footpaths of start_l stay as they are
// (td footpaths replace all static footpaths of their source).
std::vector<n::td_footpath> get_td_pareto_footpaths(
    osr::ways const& w,
    osr::lookup const& l,
    osr::platforms const& pl,
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    point_rtree<n::location_idx_t> const& loc_rtree,
    elevators const& e,
    platform_matches_t const& matches,
    n::location_idx_t const start_l,
    std::chrono::seconds const max,
    osr::bitvec<osr::node_idx_t>& blocked_mem) {
  constexpr auto const kProfile = osr::search_profile::kFoot;
  constexpr auto const kPrf = n::kFootProfile;
  blocked_mem.resize(w.n_nodes());
  assert(blocked_mem.none());

  auto const start = get_loc(tt, w, pl, matches, start_l);
  auto const [e_nodes, e_state_changes] = get_node_states(w, l, e, start.pos_);
  auto const reset =
      utl::finally{[&]() { reset_blocked(e_nodes, blocked_mem); }};

  auto const& loc = tt.locations_;
  // station graph pairs = restricted entries of the default profile
  auto const is_covered = [&](n::location_idx_t const to) {
    auto const& fps = loc.footpaths_out_[n::kDefaultProfile];
    if (start_l >= fps.size()) {
      return false;
    }
    auto const bucket = fps[start_l];
    auto const split = loc.restricted_begin(n::kDefaultProfile, true, start_l);
    for (auto i = split; i != bucket.size(); ++i) {
      if (bucket[i].target() == to) {
        return true;
      }
    }
    return false;
  };

  auto const params = to_profile_parameters(kProfile, osr_parameters{});
  auto const neighbors = get_stops_with_traffic(
      tt, rtt, loc_rtree, start,
      get_max_distance(kProfile, osr_parameters{}, max), start_l);
  auto const neighbor_locs = utl::to_vec(
      neighbors, [&](auto&& x) { return get_loc(tt, w, pl, matches, x); });

  // (target, record) -> duration per state change
  using key_t = std::pair<n::location_idx_t, n::footpath_restriction>;
  auto series = std::map<key_t, std::vector<n::duration_t>>{};
  auto reached = hash_set<n::location_idx_t>{};
  for (auto const [t_idx, change] : utl::enumerate(e_state_changes)) {
    set_blocked(e_nodes, change.second, blocked_mem);
    auto from_match = osr::match_result{};
    l.match(params, start, false, osr::direction::kForward,
            kMaxMatchingDistance, &blocked_mem, kProfile, {}, from_match);
    auto to_match = osr::match_result{};
    for (auto const& x : neighbor_locs) {
      l.match(params, x, true, osr::direction::kForward, kMaxMatchingDistance,
              &blocked_mem, kProfile, {}, to_match);
    }
    auto const results =
        osr::route_pareto(params, w, l, kProfile, start, neighbor_locs,
                          from_match[osr::match_idx_t{0U}], to_match, max,
                          osr::direction::kForward, &blocked_mem, nullptr);
    for (auto const [j, paths] : utl::enumerate(results)) {
      auto const to = neighbors[j];
      if (is_covered(to)) {
        continue;
      }
      for (auto const& p : paths) {
        reached.insert(to);
        auto& s = series[key_t{to, to_restriction(p)}];
        s.resize(e_state_changes.size(), n::kTdFootpathBan);
        s[t_idx] = std::min(
            s[t_idx], std::max(n::duration_t{1},
                               std::min(to_minutes(p), n::footpath::kMax)));
      }
    }
  }

  auto fps = std::vector<n::td_footpath>{};
  for (auto const& [key, durations] : series) {
    for (auto const [t_idx, d] : utl::enumerate(durations)) {
      fps.push_back(n::td_footpath{key.first, e_state_changes[t_idx].first, d,
                                   key.second});
    }
  }

  // Unchanged: station graph pairs (records) + footpaths not routed here.
  auto const add_static = [&](n::footpath const fp,
                              n::footpath_restriction const& r) {
    fps.push_back(n::td_footpath{fp.target(), n::kNull, fp.duration(), r});
  };
  if (start_l < loc.footpaths_out_[kPrf].size()) {
    auto const static_fps = loc.footpaths_out_[kPrf][start_l];
    auto const split = loc.restricted_begin(kPrf, true, start_l);
    for (auto i = 0U; i != static_fps.size(); ++i) {
      if (is_covered(static_fps[i].target()) ||
          !reached.contains(static_fps[i].target())) {
        add_static(
            static_fps[i],
            i < split
                ? n::footpath_restriction{}
                : loc.footpath_restrictions_out_[kPrf][start_l][i - split]);
      }
    }
  }

  utl::sort(fps);
  utl::equal_ranges_linear(
      fps,
      [](auto const& a, auto const& b) {
        return a.target_ == b.target_ && a.restriction_ == b.restriction_;
      },
      [&](std::vector<n::td_footpath>::iterator& lb,
          std::vector<n::td_footpath>::iterator& ub) {
        for (auto it = lb; it != ub; ++it) {
          if (it->duration_ == n::kTdFootpathBan && it != lb &&
              (it - 1)->duration_ != n::kTdFootpathBan) {
            it->valid_from_ -= (it - 1)->duration_ - n::duration_t{1U};
          }
        }
      });
  return fps;
}

void update_rtt_td_footpaths(osr::ways const& w,
                             osr::lookup const& l,
                             osr::platforms const& pl,
                             nigiri::timetable const& tt,
                             point_rtree<n::location_idx_t> const& loc_rtree,
                             elevators const& e,
                             platform_matches_t const& matches,
                             hash_set<n::location_idx_t> const& tasks,
                             nigiri::rt_timetable const* old_rtt,
                             nigiri::rt_timetable& rtt,
                             std::chrono::seconds const max) {
  // A footpath is time-dependent iff its source has td footpaths. Only the
  // outgoing footpaths are routed, the incoming ones are their transpose.
  auto out_mutex = std::mutex{};
  auto out = std::map<n::location_idx_t, std::vector<n::td_footpath>>{};
  utl::parallel_for_run_threadlocal<osr::bitvec<osr::node_idx_t>>(
      tasks.size(),
      [&](osr::bitvec<osr::node_idx_t>& blocked, std::size_t const task_idx) {
        auto const start = *(begin(tasks) + task_idx);
        auto fps = get_td_pareto_footpaths(w, l, pl, tt, &rtt, loc_rtree, e,
                                           matches, start, max, blocked);
        {
          auto const lock = std::unique_lock{out_mutex};
          out[start] = std::move(fps);
        }
      });

  // Routed between stops: virtual locations walk like their stop.
  n::expand_td_footpaths_to_virts(tt, out);

  // The td footpaths of a source replace all its static footpaths:
  // has_td_footpaths_in_ marks the targets of both.
  auto in = n::vector_map<n::location_idx_t, std::vector<n::td_footpath>>{};
  in.resize(tt.n_locations());
  rtt.has_td_footpaths_in_[n::kFootProfile].zero_out();
  rtt.td_footpaths_out_[n::kFootProfile].clear();
  for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
    auto const it = out.find(i);
    if (it != end(out)) {
      rtt.has_td_footpaths_out_[n::kFootProfile].set(i, true);
      rtt.td_footpaths_out_[n::kFootProfile].emplace_back(it->second);
    } else if (old_rtt != nullptr) {
      rtt.has_td_footpaths_out_[n::kFootProfile].set(
          i, old_rtt->has_td_footpaths_out_[n::kFootProfile].test(i));
      rtt.td_footpaths_out_[n::kFootProfile].emplace_back(
          old_rtt->td_footpaths_out_[n::kFootProfile][i]);
    } else {
      rtt.has_td_footpaths_out_[n::kFootProfile].set(i, false);
      rtt.td_footpaths_out_[n::kFootProfile].emplace_back(
          std::initializer_list<n::td_footpath>{});
    }

    if (!rtt.has_td_footpaths_out_[n::kFootProfile].test(i)) {
      continue;
    }
    for (auto const& fp : rtt.td_footpaths_out_[n::kFootProfile][i]) {
      in[fp.target_].push_back(
          n::td_footpath{i, fp.valid_from_, fp.duration_, fp.restriction_});
      rtt.has_td_footpaths_in_[n::kFootProfile].set(fp.target_, true);
    }
    for (auto const& fp : tt.locations_.footpaths_out_[n::kFootProfile][i]) {
      rtt.has_td_footpaths_in_[n::kFootProfile].set(fp.target(), true);
    }
  }

  rtt.td_footpaths_in_[n::kFootProfile].clear();
  for (auto const& fps : in) {
    rtt.td_footpaths_in_[n::kFootProfile].emplace_back(fps);
  }
}

}  // namespace motis
