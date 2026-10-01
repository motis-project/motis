#include "motis/flex/flex.h"

#include <memory>
#include <optional>
#include <ranges>

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "nigiri/logging.h"

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
#include "motis/osr/one_to_many_searches.h"

namespace n = nigiri;

namespace motis::flex {

void verify_flex_limits(n::timetable const& tt) {
  constexpr auto const kMaxTransports = 1U << mode_payload::kTransportBits;
  constexpr auto const kMaxStops = 1U << mode_payload::kStopBits;
  auto const n_transports =
      static_cast<std::size_t>(tt.flex_transport_stop_seq_.size());
  auto max_stops = std::size_t{0U};
  for (auto const t : tt.flex_transport_stop_seq_) {
    max_stops = std::max(
        max_stops, static_cast<std::size_t>(tt.flex_stop_seq_[t].size()));
  }
  auto const pct = [](std::size_t const n, std::size_t const limit) {
    return 100.0 * static_cast<double>(n) / static_cast<double>(limit);
  };
  n::log(n::log_lvl::info, "motis.flex",
         "flex transports: {} of {} ({:.2f}%), "
         "most stop rows in a flex transport: {} of {} ({:.2f}%)",
         n_transports, kMaxTransports, pct(n_transports, kMaxTransports),
         max_stops, kMaxStops, pct(max_stops, kMaxStops));
  utl::verify(n_transports <= kMaxTransports && max_stops <= kMaxStops,
              "flex data exceeds system limitations: {} flex transports "
              "(limit {}), {} stop rows in one flex transport (limit {})",
              n_transports, kMaxTransports, max_stops, kMaxStops);
}

osr::sharing_data prepare_sharing_data(n::timetable const& tt,
                                       osr::ways const& w,
                                       osr::lookup const& lookup,
                                       osr::platforms const* pl,
                                       flex_areas const& fa,
                                       platform_matches_t const* pl_matches,
                                       mode_payload const id,
                                       flex_routing_data& frd) {
  // Start / end in travel order, independent of the search direction: osr's
  // car_sharing profile reads start_allowed_ / end_allowed_ that way.
  auto const stop_seq =
      tt.flex_stop_seq_[tt.flex_transport_stop_seq_[id.get_flex_transport()]];
  auto const from_stop = stop_seq.at(id.get_from_stop());
  auto const to_stop = stop_seq.at(id.get_to_stop());

  // Count additional nodes and allocate bit vectors.
  auto n_nodes = w.n_nodes();
  for (auto const& s : {from_stop, to_stop}) {
    s.apply(utl::overloaded{[&](n::location_group_idx_t const lg) {
      n_nodes += tt.location_group_locations_[lg].size();
    }});
  }
  frd.additional_nodes_.reset(w.n_nodes());
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
    frd.additional_nodes_.locations_.emplace_back(l);
    frd.additional_nodes_.coordinates_.emplace_back(
        tt.locations_.coordinates_[l]);

    auto const pos = get_location(&tt, &w, pl, pl_matches, tt_location{l});
    auto const l_additional_node_idx = next_add_node_idx++;

    auto matches = osr::match_result{};
    lookup.complete_match<osr::foot<false>>(
        osr::foot<false>::parameters{}, pos, false, osr::direction::kForward,
        kMaxGbfsMatchingDistance, nullptr, std::nullopt, {}, matches);

    auto const m = matches[osr::match_idx_t{0U}];
    for (auto j = std::size_t{0U}; j != m.size(); ++j) {
      auto const handle_node = [&](osr::candidate_node const& node) {
        if (!node.valid() || node.dist_to_node_ > kMaxGbfsMatchingDistance) {
          return;
        }

        auto const edge_to_an = osr::additional_edge{
            l_additional_node_idx,
            static_cast<osr::distance_t>(node.dist_to_node_)};
        auto& node_edges = frd.additional_nodes_.edges_[node.node_];
        if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
          node_edges.emplace_back(edge_to_an);
        }

        auto& add_node_out =
            frd.additional_nodes_.edges_[l_additional_node_idx];
        auto const edge_from_an = osr::additional_edge{
            node.node_, static_cast<osr::distance_t>(node.dist_to_node_)};
        if (utl::find(add_node_out, edge_from_an) == end(add_node_out)) {
          add_node_out.emplace_back(edge_from_an);
        }
      };

      handle_node(m.left(j));
      handle_node(m.right(j));
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

  // Set end allowed in the alighting area / location group.
  to_stop.apply(utl::overloaded{
      [&](n::location_group_idx_t const to_lg) {
        for (auto const& l : tt.location_group_locations_[to_lg]) {
          frd.end_allowed_.set(add_tt_location(l), true);
        }
      },
      [&](n::flex_area_idx_t const to_area) {
        fa.add_area(to_area, frd.end_allowed_, tmp);
      }});

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
    std::chrono::seconds const max,
    osr_parameters const& osr_params) {
  auto routings = flex_routings_t{};

  // Traffic days helpers.
  auto const day_idx_iv = get_relevant_days(tt, start_time);
  auto const is_active = [&](n::flex_transport_idx_t const t) {
    auto const& bitfield = tt.bitfields_[tt.flex_transport_traffic_days_[t]];
    return utl::any_of(day_idx_iv, [&](n::day_idx_t const i) {
      return bitfield.test(to_idx(i));
    });
  };

  // Adds one routing per (boarding, alighting) stop pair of transport `t`
  // in which the query position's stop `x` takes part: as boarding stop
  // for the forward search (first mile, the ride starts at the position), as
  // alighting stop for the backward search (last mile, the ride ends there).
  auto const add_flex_transport = [&](n::flex_transport_idx_t const t,
                                      n::flex_stop_t const x) {
    if (!is_active(t)) {
      return;
    }
    auto const stop_seq_idx = tt.flex_transport_stop_seq_[t];
    auto const stops = tt.flex_stop_seq_[stop_seq_idx];
    auto const add = [&](n::stop_idx_t const from, n::stop_idx_t const to) {
      if (mode_payload::fits(t, from, to)) {
        routings[std::pair{stop_seq_idx, std::pair{from, to}}].emplace_back(
            t, from, to);
      }
    };
    for (auto i = n::stop_idx_t{0U}; i != stops.size(); ++i) {
      if (!(stops[i] == x)) {
        continue;
      }
      if (dir == osr::direction::kForward) {
        for (auto j = static_cast<n::stop_idx_t>(i + 1U); j < stops.size();
             ++j) {
          add(i, j);
        }
      } else {
        for (auto j = n::stop_idx_t{0U}; j != i; ++j) {
          add(j, i);
        }
      }
    }
  };

  // Collect area transports.
  auto const add_area_flex_transports = [&](n::flex_area_idx_t const a) {
    for (auto const t : tt.flex_area_transports_[a]) {
      add_flex_transport(t, a);
    }
  };
  auto const box = geo::box{
      pos, get_max_distance(osr::search_profile::kFoot, osr_params, max)};
  tt.flex_area_rtree_.search(box.min_.lnglat_float(), box.max_.lnglat_float(),
                             [&](auto&&, auto&&, n::flex_area_idx_t const a) {
                               add_area_flex_transports(a);
                               return true;
                             });

  // Collect location group transports.
  auto location_groups = hash_set<n::location_group_idx_t>{};
  loc_rtree.in_radius(
      pos, get_max_distance(osr::search_profile::kFoot, osr_params, max),
      [&](n::location_idx_t const l) {
        for (auto const lg : tt.location_location_groups_[l]) {
          location_groups.emplace(lg);
        }
        return true;
      });
  for (auto const& lg : location_groups) {
    for (auto const t : tt.location_group_transports_[lg]) {
      add_flex_transport(t, lg);
    }
  }

  return routings;
}

n::interval<n::unixtime_t> get_departure_window(n::timetable const& tt,
                                                mode_payload const id,
                                                n::unixtime_t const day,
                                                n::duration_t const duration) {
  auto const windows =
      tt.flex_transport_stop_time_windows_[id.get_flex_transport()];
  auto const from = windows[id.get_from_stop()];
  auto const to = windows[id.get_to_stop()];
  return n::interval{day + from.from_, day + from.to_}.intersect(
      n::interval{day + to.from_ - duration, day + to.to_ - duration});
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
                         std::map<std::string, std::uint64_t>& stats,
                         one_to_many_side* const states) {
  UTL_START_TIMING(flex_lookup_timer);

  auto const max_dist =
      get_max_distance(osr::search_profile::kCarSharing, osr_params, max);
  auto const near_stops = loc_rtree.in_radius(pos.pos_, max_dist);
  auto const near_stop_locations =
      utl::to_vec(near_stops, [&](n::location_idx_t const l) {
        return get_location(&tt, &w, pl, matches, tt_location{l});
      });

  auto const params =
      to_profile_parameters(osr::search_profile::kCarSharing, osr_params);
  auto pos_match = osr::match_result{};
  lookup.match(params, pos, false, dir, max_matching_distance, nullptr,
               osr::search_profile::kCarSharing, {}, pos_match);
  auto const near_stop_matches = get_reverse_platform_way_matches(
      lookup, way_matches, osr::search_profile::kCarSharing, near_stops,
      near_stop_locations, dir, max_matching_distance);

  auto const routings = get_flex_routings(tt, loc_rtree, start_time, pos.pos_,
                                          dir, max, osr_params);

  stats.emplace(fmt::format("prepare_{}_FLEX_lookup", to_str(dir)),
                UTL_GET_TIMING_MS(flex_lookup_timer));

  for (auto const& [stop_seq, transports] : routings) {
    UTL_START_TIMING(routing_timer);

    auto const sharing_data = prepare_sharing_data(
        tt, w, lookup, pl, fa, matches, transports.front(), frd);

    auto state = osr::route_one_to_many(
        params, w, lookup, osr::search_profile::kCarSharing, pos,
        near_stop_locations, pos_match[osr::match_idx_t{0U}], near_stop_matches,
        static_cast<osr::cost_t>(max.count()), dir, nullptr, &sharing_data,
        nullptr);
    auto const& paths = state->results();

    // Store osr routing state for later path reconstruction.
    if (states != nullptr) {
      auto dest_idx = hash_map<n::location_idx_t, std::size_t>{};
      for (auto const [i, l] : utl::enumerate(near_stops)) {
        if (paths[i].has_value()) {
          dest_idx.emplace(l, i);
        }
      }

      if (!dest_idx.empty()) {
        auto const search_idx = states->searches_.size();
        states->searches_.emplace_back(
            one_to_many_search{std::move(state), std::move(dest_idx),
                               std::move(frd.additional_nodes_)});

        for (auto const id : transports) {
          states
              ->by_mode_[transport_mode(api::ModeEnum::FLEX, id.to_payload())] =
              search_idx;
        }
      }
    }

    // The offsets are indexed by the departure time in travel direction (the
    // start of the whole access / egress), in both search directions: that is
    // how nigiri's td lookup reads them.
    auto const day_idx_iv = get_relevant_days(tt, start_time);
    for (auto const id : transports) {
      auto const t = id.get_flex_transport();
      for (auto const day_idx : day_idx_iv) {
        if (!tt.bitfields_[tt.flex_transport_traffic_days_[t]].test(
                to_idx(day_idx))) {
          continue;
        }

        auto const day =
            tt.internal_interval().from_ + to_idx(day_idx) * date::days{1U};
        for (auto const [p, l] : utl::zip(paths, near_stops)) {
          if (p.has_value()) {
            auto const duration = n::duration_t{p->cost_ / 60};
            auto const dep_iv = get_departure_window(tt, id, day, duration);

            if (dep_iv.from_ < dep_iv.to_ &&
                duration < n::footpath::kMaxDuration) {
              auto const mode =
                  transport_mode(api::ModeEnum::FLEX, id.to_payload());
              auto& offsets = ret[l];
              offsets.push_back(
                  n::routing::td_offset::make(dep_iv.from_, duration, mode));
              offsets.push_back(n::routing::td_offset::make(
                  dep_iv.to_, n::footpath::kMaxDuration, mode));
            }
          }
        }
      }
    }

    auto const& [seq_idx, stop_pair] = stop_seq;
    stats.emplace(
        fmt::format("prepare_{}_FLEX_{}", to_str(dir),
                    tt.flex_stop_seq_[seq_idx][stop_pair.first].apply(
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
}

}  // namespace motis::flex
