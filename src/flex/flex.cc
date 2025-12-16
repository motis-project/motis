#include "motis/flex/flex.h"

#include <ranges>

#include "utl/concat.h"

#include "osr/lookup.h"
#include "osr/routing/parameters.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"
#include "osr/ways.h"

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/flex/flex_areas.h"
#include "motis/flex/flex_routing_data.h"
#include "motis/match_platforms.h"
#include "motis/osr/max_distance.h"

namespace n = nigiri;

namespace motis::flex {

osr::sharing_data prepare_sharing_data(n::timetable const& tt,
                                       osr::ways const& w,
                                       osr::lookup const& lookup,
                                       osr::platforms const* pl,
                                       flex_areas const& fa,
                                       platform_matches_t const* pl_matches,
                                       mode_id const id,
                                       osr::direction const dir,
                                       flex_routing_data& frd) {
  auto const stop_seq =
      tt.flex_stop_seq_[tt.flex_transport_stop_seq_[id.get_flex_transport()]];
  auto const from_stop = stop_seq.at(id.get_stop());
  auto to_stops = std::vector<n::flex_stop_t>{};
  for (auto i = static_cast<int>(id.get_stop()) +
                (dir == osr::direction::kForward ? 1 : -1);
       dir == osr::direction::kForward ? i < static_cast<int>(stop_seq.size())
                                       : i >= 0;
       dir == osr::direction::kForward ? ++i : --i) {
    to_stops.emplace_back(stop_seq.at(static_cast<n::stop_idx_t>(i)));
  }

  // Count additional nodes and allocate bit vectors.
  auto n_nodes = w.n_nodes();
  from_stop.apply(utl::overloaded{[&](n::location_group_idx_t const from_lg) {
    n_nodes += tt.location_group_locations_[from_lg].size();
  }});
  for (auto const& to_stop : to_stops) {
    to_stop.apply(utl::overloaded{[&](n::location_group_idx_t const to_lg) {
      n_nodes += tt.location_group_locations_[to_lg].size();
    }});
  }
  frd.additional_node_offset_ = w.n_nodes();
  frd.additional_node_coordinates_.clear();
  frd.additional_edges_.clear();
  frd.start_allowed_.resize(n_nodes);
  frd.end_allowed_.resize(n_nodes);
  frd.through_allowed_.resize(n_nodes);
  frd.start_allowed_.zero_out();
  frd.end_allowed_.zero_out();
  frd.through_allowed_.one_out();

  // Creates an additional node for the given timetable location
  // and adds additional edges to/from this node.
  auto next_add_node_idx = osr::node_idx_t{w.n_nodes()};
  auto const add_tt_location = [&](n::location_idx_t const l) {
    frd.additional_nodes_.emplace_back(l);
    frd.additional_node_coordinates_.emplace_back(
        tt.locations_.coordinates_[l]);

    auto const pos = get_location(&tt, &w, pl, pl_matches, tt_location{l});
    auto const l_additional_node_idx = next_add_node_idx++;

    auto const matches = lookup.match<osr::foot<false>>(
        osr::foot<false>::parameters{}, pos, false, osr::direction::kForward,
        kMaxGbfsMatchingDistance, nullptr);

    for (auto const& m : matches) {
      auto const handle_node = [&](osr::node_candidate const& node) {
        if (!node.valid() || node.dist_to_node_ > kMaxGbfsMatchingDistance) {
          return;
        }

        auto const edge_to_an = osr::additional_edge{
            l_additional_node_idx,
            static_cast<osr::distance_t>(node.dist_to_node_)};
        auto& node_edges = frd.additional_edges_[node.node_];
        if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
          node_edges.emplace_back(edge_to_an);
        }

        auto& add_node_out = frd.additional_edges_[l_additional_node_idx];
        auto const edge_from_an = osr::additional_edge{
            node.node_, static_cast<osr::distance_t>(node.dist_to_node_)};
        if (utl::find(add_node_out, edge_from_an) == end(add_node_out)) {
          add_node_out.emplace_back(edge_from_an);
        }
      };

      handle_node(m.left_);
      handle_node(m.right_);
    }

    return l_additional_node_idx;
  };

  // Set start allowed in start area / location group.
  auto tmp = osr::bitvec<osr::node_idx_t>{};
  from_stop.apply(utl::overloaded{
      [&](n::location_group_idx_t const from_lg) {
        for (auto const& l : tt.location_group_locations_[from_lg]) {
          frd.start_allowed_.set(add_tt_location(l), true);
        }
      },
      [&](n::flex_area_idx_t const from_area) {
        fa.add_area(from_area, frd.start_allowed_, tmp);
      }});

  // Set end allowed in follow-up areas / location groups.
  for (auto const& to_stop : to_stops) {
    to_stop.apply(utl::overloaded{
        [&](n::location_group_idx_t const to_lg) {
          for (auto const& l : tt.location_group_locations_[to_lg]) {
            frd.end_allowed_.set(add_tt_location(l), true);
          }
        },
        [&](n::flex_area_idx_t const to_area) {
          fa.add_area(to_area, frd.end_allowed_, tmp);
        }});
  }

  return frd.to_sharing_data();
}

n::interval<n::day_idx_t> get_relevant_days(
    n::timetable const& tt, n::routing::start_time_t const start_time) {
  auto const to_sys_days = [](n::unixtime_t const t) {
    return std::chrono::time_point_cast<date::sys_days::duration>(t);
  };
  auto const iv = std::visit(
      utl::overloaded{[&](n::unixtime_t const t) {
                        return n::interval{to_sys_days(t) - date::days{2},
                                           to_sys_days(t) + date::days{3}};
                      },
                      [&](n::interval<n::unixtime_t> const x) {
                        return n::interval{to_sys_days(x.from_) - date::days{2},
                                           to_sys_days(x.to_) + date::days{3}};
                      }},
      start_time);
  return n::interval{tt.day_idx(iv.from_), tt.day_idx(iv.to_)};
}

flex_routings_t get_flex_routings(
    n::timetable const& tt,
    point_rtree<n::location_idx_t> const& loc_rtree,
    n::routing::start_time_t const start_time,
    geo::latlng const& pos,
    osr::direction const dir,
    std::chrono::seconds const max) {
  auto routings = flex_routings_t{};

  // Traffic days helpers.
  auto const day_idx_iv = get_relevant_days(tt, start_time);
  auto const is_active = [&](n::flex_transport_idx_t const t) {
    auto const& bitfield = tt.bitfields_[tt.flex_transport_traffic_days_[t]];
    return utl::any_of(day_idx_iv, [&](n::day_idx_t const i) {
      return bitfield.test(to_idx(i));
    });
  };

  // Stop index helper.
  auto const get_stop_idx =
      [&](n::flex_stop_seq_idx_t const stop_seq_idx,
          n::flex_stop_t const x) -> std::optional<n::stop_idx_t> {
    auto const stops = tt.flex_stop_seq_[stop_seq_idx];
    auto const is_last = [&](n::stop_idx_t const stop_idx) {
      return (dir == osr::direction::kBackward && stop_idx == 0U) ||
             (dir == osr::direction::kForward && stop_idx == stops.size() - 1U);
    };
    for (auto c = 0U; c != stops.size(); ++c) {
      auto const stop_idx = static_cast<n::stop_idx_t>(
          dir == osr::direction::kForward ? c : stops.size() - c - 1);
      if (stops[stop_idx] == x && !is_last(stop_idx)) {
        return stop_idx;
      }
    }
    return std::nullopt;
  };

  // Collect area transports.
  auto const add_area_flex_transports = [&](n::flex_area_idx_t const a) {
    for (auto const t : tt.flex_area_transports_[a]) {
      if (!is_active(t)) {
        continue;
      }

      auto const stop_idx = get_stop_idx(tt.flex_transport_stop_seq_[t], a);
      if (stop_idx.has_value()) {
        routings[std::pair{tt.flex_transport_stop_seq_[t], *stop_idx}]
            .emplace_back(t, *stop_idx, dir);
      }
    }
  };
  auto const box =
      geo::box{pos, get_max_distance(osr::search_profile::kFoot, max)};
  tt.flex_area_rtree_.search(box.min_.lnglat_float(), box.max_.lnglat_float(),
                             [&](auto&&, auto&&, n::flex_area_idx_t const a) {
                               add_area_flex_transports(a);
                               return true;
                             });

  // Collect location group transports.
  auto location_groups = hash_set<n::location_group_idx_t>{};
  loc_rtree.in_radius(pos, get_max_distance(osr::search_profile::kFoot, max),
                      [&](n::location_idx_t const l) {
                        for (auto const lg : tt.location_location_groups_[l]) {
                          location_groups.emplace(lg);
                        }
                        return true;
                      });
  for (auto const& lg : location_groups) {
    for (auto const t : tt.location_group_transports_[lg]) {
      if (!is_active(t)) {
        continue;
      }

      auto const stop_idx = get_stop_idx(tt.flex_transport_stop_seq_[t], lg);
      if (stop_idx.has_value()) {
        routings[std::pair{tt.flex_transport_stop_seq_[t], *stop_idx}]
            .emplace_back(t, *stop_idx, dir);
      }
    }
  }

  return routings;
}

bool is_in_flex_stop(n::timetable const& tt,
                     osr::ways const& w,
                     flex_areas const& fa,
                     flex_routing_data const& frd,
                     n::flex_stop_t const& s,
                     osr::node_idx_t const n) {
  return s.apply(utl::overloaded{
      [&](n::flex_area_idx_t const a) {
        return !w.is_additional_node(n) && n != osr::node_idx_t::invalid() &&
               fa.is_in_area(a, w.get_node_pos(n));
      },
      [&](n::location_group_idx_t const lg) {
        if (!w.is_additional_node(n)) {
          return false;
        }
        auto const locations = tt.location_group_locations_.at(lg);
        auto const l = frd.get_additional_node(n);
        return utl::find(locations, l) != end(locations);
      }});
}

void add_flex_td_offsets(osr::ways const& w,
                         osr::lookup const& lookup,
                         osr::platforms const* pl,
                         platform_matches_t const* matches,
                         way_matches_storage const* way_matches,
                         n::timetable const& tt,
                         flex_areas const& fa,
                         point_rtree<n::location_idx_t> const& loc_rtree,
                         n::routing::start_time_t const start_time,
                         osr::location const& pos,
                         osr::direction const dir,
                         std::chrono::seconds const max,
                         double const max_matching_distance,
                         osr_parameters const& osr_params,
                         flex_routing_data& frd,
                         n::routing::td_offsets_t& ret,
                         std::map<std::string, std::uint64_t>& stats) {
  UTL_START_TIMING(flex_lookup_timer);

  auto const max_dist = get_max_distance(osr::search_profile::kCarSharing, max);
  auto const near_stops = loc_rtree.in_radius(pos.pos_, max_dist);
  auto const near_stop_locations =
      utl::to_vec(near_stops, [&](n::location_idx_t const l) {
        return get_location(&tt, &w, pl, matches, tt_location{l});
      });

  auto const params =
      to_profile_parameters(osr::search_profile::kCarSharing, osr_params);
  auto const pos_match =
      lookup.match(params, pos, false, dir, max_matching_distance, nullptr,
                   osr::search_profile::kCarSharing);
  auto const near_stop_matches = get_reverse_platform_way_matches(
      lookup, way_matches, osr::search_profile::kCarSharing, near_stops,
      near_stop_locations, dir, max_matching_distance);

  auto const routings =
      get_flex_routings(tt, loc_rtree, start_time, pos.pos_, dir, max);

  stats.emplace(fmt::format("prepare_{}_FLEX_lookup", to_str(dir)),
                UTL_GET_TIMING_MS(flex_lookup_timer));

  for (auto const& [stop_seq, transports] : routings) {
    UTL_START_TIMING(routing_timer);

    auto const sharing_data = prepare_sharing_data(
        tt, w, lookup, pl, fa, matches, transports.front(), dir, frd);

    auto const paths =
        osr::route(params, w, lookup, osr::search_profile::kCarSharing, pos,
                   near_stop_locations, pos_match, near_stop_matches,
                   static_cast<osr::cost_t>(max.count()), dir, nullptr,
                   &sharing_data, nullptr);
    auto const day_idx_iv = get_relevant_days(tt, start_time);
    for (auto const id : transports) {
      auto const t = id.get_flex_transport();
      auto const from_stop_idx = id.get_stop();

      for (auto const day_idx : day_idx_iv) {
        if (!tt.bitfields_[tt.flex_transport_traffic_days_[t]].test(
                to_idx(day_idx))) {
          continue;
        }

        auto const day =
            tt.internal_interval().from_ + to_idx(day_idx) * date::days{1U};
        auto const from_stop_time_window =
            tt.flex_transport_stop_time_windows_[t][from_stop_idx];
        auto const abs_from_stop_iv = n::interval{
            day + from_stop_time_window.from_, day + from_stop_time_window.to_};
        for (auto const [p, s, l] :
             utl::zip(paths, near_stop_locations, near_stops)) {
          if (p.has_value()) {
            auto const rel_to_stop_idx = 0U;
            auto const to_stop_idx = static_cast<n::stop_idx_t>(
                dir == osr::direction::kForward
                    ? from_stop_idx + rel_to_stop_idx
                    : from_stop_idx - rel_to_stop_idx);
            auto const duration = n::duration_t{p->cost_ / 60};
            auto const to_stop_time_window =
                tt.flex_transport_stop_time_windows_[t][to_stop_idx];
            auto const abs_to_stop_iv = n::interval{
                day + to_stop_time_window.from_, day + to_stop_time_window.to_};

            auto const iv_at_to_stop =
                (dir == osr::direction::kForward ? abs_from_stop_iv >> duration
                                                 : abs_from_stop_iv << duration)
                    .intersect(abs_to_stop_iv);
            auto const iv_at_from_stop = dir == osr::direction::kForward
                                             ? iv_at_to_stop << duration
                                             : iv_at_to_stop >> duration;

            auto& offsets = ret[l];
            if (offsets.empty()) {
              offsets.emplace_back(n::unixtime_t{n::i32_minutes{0U}},
                                   n::footpath::kMaxDuration, id.to_id());
            }
            offsets.emplace_back(iv_at_from_stop.from_, duration, id.to_id());
            offsets.emplace_back(iv_at_from_stop.to_, n::footpath::kMaxDuration,
                                 id.to_id());
          }
        }
      }
    }

    stats.emplace(
        fmt::format("prepare_{}_FLEX_{}", to_str(dir),
                    tt.flex_stop_seq_[stop_seq.first][stop_seq.second].apply(
                        utl::overloaded{[&](n::location_group_idx_t const g) {
                                          return tt.get_default_translation(
                                              tt.location_group_name_[g]);
                                        },
                                        [&](n::flex_area_idx_t const a) {
                                          return tt.get_default_translation(
                                              tt.flex_area_name_[a]);
                                        }})),
        UTL_GET_TIMING_MS(routing_timer));
  }

  for (auto& [_, offsets] : ret) {
    utl::sort(offsets, [](n::routing::td_offset const& a,
                          n::routing::td_offset const& b) {
      return a.valid_from_ < b.valid_from_;
    });
  }
}

}  // namespace motis::flex
