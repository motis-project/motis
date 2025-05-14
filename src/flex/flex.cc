#include "motis/flex/flex.h"

#include <ranges>

#include "nigiri/flex.h"

#include "utl/concat.h"

#include "osr/lookup.h"
#include "osr/routing/profiles/foot.h"
#include "osr/ways.h"

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/flex/flex_routing_data.h"
#include "motis/max_distance.h"

namespace n = nigiri;

namespace motis::flex {

template <typename Fn>
void for_each_area_node(n::timetable const& tt,
                        osr::ways const& w,
                        osr::lookup const& l,
                        n::flex_area_idx_t const a,
                        Fn&& fn) {
  l.find(tt.flex_area_bbox_[a], [&](osr::way_idx_t const way) {
    for (auto const& x : w.r_->way_nodes_[way]) {
      if (n::is_within(tt, a, w.get_node_pos(x))) {
        fn(x);
      }
    }
  });
}

osr::sharing_data prepare_sharing_data(n::timetable const& tt,
                                       osr::ways const& w,
                                       osr::lookup const& lookup,
                                       osr::platforms const* pl,
                                       platform_matches_t const* pl_matches,
                                       mode_id const id,
                                       osr::direction const dir,
                                       flex_routing_data& frd) {
  auto const stop_seq = tt.flex_transport_stop_seq_[id.get_flex_transport()];
  auto const from_stop = stop_seq.at(id.get_stop());
  auto to_stops = std::vector<n::flex_stop_t>{};
  for (auto i = static_cast<int>(id.get_stop());
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

    auto const matches =
        lookup.match<osr::foot<false>>(pos, false, osr::direction::kForward,
                                       kMaxGbfsMatchingDistance, nullptr);

    auto& add_node_out = frd.additional_edges_[l_additional_node_idx];
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
  from_stop.apply(utl::overloaded{
      [&](n::location_group_idx_t const from_lg) {
        for (auto const& l : tt.location_group_locations_[from_lg]) {
          frd.start_allowed_.set(add_tt_location(l), true);
        }
      },
      [&](n::flex_area_idx_t const from_area) {
        for_each_area_node(
            tt, w, lookup, from_area,
            [&](osr::node_idx_t const n) { frd.start_allowed_.set(n, true); });
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
          for_each_area_node(
              tt, w, lookup, to_area,
              [&](osr::node_idx_t const n) { frd.end_allowed_.set(n, true); });
        }});
  }

  return frd.to_sharing_data();
}

void add_flex_td_offsets(ep::routing const& r,
                         osr::location const& pos,
                         osr::direction const dir,
                         double const,
                         std::chrono::seconds const max,
                         nigiri::routing::start_time_t const& start_time,
                         n::routing::td_offsets_t&) {
  auto const& tt = *r.tt_;
  auto routings = hash_map<
      std::pair<n::flex_stop_t, std::vector<n::flex_stop_t>>,
      std::vector<std::pair<n::flex_transport_idx_t, n::stop_idx_t>>>{};

  // ============
  // Traffic days
  // ------------
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
  auto const day_idx_iv = n::interval{tt.day_idx(iv.from_), tt.day_idx(iv.to_)};
  auto const get_traffic_days = [&](n::flex_transport_idx_t const t) {
    return tt.bitfields_[tt.flex_transport_traffic_days_[t]];
  };
  auto const is_active = [&](n::flex_transport_idx_t const t) {
    auto const& bitfield = get_traffic_days(t);
    return utl::any_of(day_idx_iv, [&](n::day_idx_t const i) {
      return bitfield.test(to_idx(i));
    });
  };

  // =======================
  // Collect area transports
  // -----------------------
  auto const get_next_stops = [&](auto const& stops, n::flex_stop_t const x)
      -> std::pair<std::vector<n::flex_stop_t>, int> {
    auto next_stops = std::vector<n::flex_stop_t>{};
    auto area_seen = false;
    auto first_stop_idx = -1;
    for (auto c = 0U; c != stops.size(); ++c) {
      auto const stop_idx = static_cast<n::stop_idx_t>(
          dir == osr::direction::kForward ? c : stops.size() - c - 1);
      auto const stop = stops[stop_idx];

      if (area_seen) {
        next_stops.emplace_back(stop);
      }

      if (stop == x) {
        area_seen = true;
        first_stop_idx = static_cast<int>(stop_idx);
      }
    }
    return {next_stops, first_stop_idx};
  };

  auto const add_area_flex_transports = [&](n::flex_area_idx_t const a) {
    for (auto const t : tt.flex_area_transports_[a]) {
      auto [next_stops, first_stop_idx] =
          get_next_stops(tt.flex_transport_stop_seq_[t], a);
      utl::verify(first_stop_idx != -1, "first stop not found");
      if (next_stops.empty() || !is_active(t)) {
        continue;
      }
      routings[{a, std::move(next_stops)}].emplace_back(t, first_stop_idx);
    }
  };
  tt.flex_area_rtree_.search(pos.pos_.lnglat_float(), pos.pos_.lnglat_float(),
                             [&](auto&&, auto&&, n::flex_area_idx_t const a) {
                               if (n::is_within(tt, a, pos.pos_)) {
                                 add_area_flex_transports(a);
                               }
                               return true;
                             });

  // =================================
  // Collect location group transports
  // ---------------------------------
  auto location_groups = hash_set<n::location_group_idx_t>{};
  r.loc_tree_->in_radius(
      pos.pos_, get_max_distance(osr::search_profile::kFoot, max),
      [&](n::location_idx_t const l) {
        for (auto const lg : tt.location_location_groups_[l]) {
          location_groups.emplace(lg);
        }
        return true;
      });
  for (auto const& lg : location_groups) {
    for (auto const t : tt.location_group_transports_[lg]) {
      auto [next_stops, first_stop_idx] =
          get_next_stops(tt.flex_transport_stop_seq_[t], lg);
      utl::verify(first_stop_idx != -1, "first stop not found");
      if (next_stops.empty() || !is_active(t)) {
        continue;
      }
      routings[{lg, next_stops}].emplace_back(t, first_stop_idx);
    }
  }

  // =======
  // Routing
  // -------
  //  auto frd = flex_routing_data{};
  //  for (auto const& [from_stop_follow_up_stops, transports] : routings) {
  //    auto const& [from_stop, follow_up_stops] = from_stop_follow_up_stops;
  //
  //    auto const max_dist =
  //        get_max_distance(osr::search_profile::kCarSharing, max);
  //    auto const near_stops = r.loc_tree_->in_radius(pos.pos_, max_dist);
  //    auto const near_stop_locations =
  //        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
  //          return get_location(r.tt_, r.w_, r.pl_, r.matches_,
  //          tt_location{l});
  //        });
  //
  //    auto const sharing_data = prepare_sharing_data(
  //        tt, *r.w_, *r.l_, r.pl_, r.matches_, from_stop, follow_up_stops,
  //        frd);
  //    auto const paths =
  //        osr::route(*r.w_, *r.l_, osr::search_profile::kCarSharing, pos,
  //                   near_stop_locations,
  //                   static_cast<osr::cost_t>(max.count()), dir,
  //                   max_matching_distance, nullptr, &sharing_data, nullptr);
  //
  //    for (auto const day_idx : day_idx_iv) {
  //      for (auto const& [t, from_stop_idx] : transports) {
  //        if (!get_traffic_days(t).test(to_idx(day_idx))) {
  //          continue;
  //        }
  //
  //        auto const day =
  //            tt.internal_interval().from_ + to_idx(day_idx) * date::days{1U};
  //        auto const from_stop_time_window =
  //            tt.flex_transport_stop_time_windows_[t][from_stop_idx];
  //        auto const abs_from_stop_iv = n::interval{
  //            day + from_stop_time_window.from_, day +
  //            from_stop_time_window.to_};
  //
  //        for (auto const [p, s] : utl::zip(paths, near_stop_locations)) {
  //          if (p.has_value() && p->track_node_ != osr::node_idx_t::invalid())
  //          {
  //            auto const [l, rel_to_stop_idx] = s;
  //            auto const to_stop_idx = static_cast<n::stop_idx_t>(
  //                dir == osr::direction::kForward
  //                    ? from_stop_idx + rel_to_stop_idx
  //                    : from_stop_idx - rel_to_stop_idx);
  //            auto const duration = n::duration_t{p->cost_ / 60};
  //            auto const mode_id =
  //                static_cast<n::transport_mode_id_t>(0);  // TODO
  //
  //            auto const to_stop_time_window =
  //                tt.flex_transport_stop_time_windows_[t][to_stop_idx];
  //            auto const abs_to_stop_iv = n::interval{
  //                day + to_stop_time_window.from_, day +
  //                to_stop_time_window.to_};
  //
  //            auto const iv_at_to_stop =
  //                (dir == osr::direction::kForward ? abs_from_stop_iv >>
  //                duration
  //                                                 : abs_from_stop_iv <<
  //                                                 duration)
  //                    .intersect(abs_to_stop_iv);
  //            auto const iv_at_from_stop = dir == osr::direction::kForward
  //                                             ? iv_at_to_stop << duration
  //                                             : iv_at_to_stop >> duration;
  //
  //            auto& offsets = ret[l];
  //            if (offsets.empty()) {
  //              offsets.emplace_back(n::unixtime_t{n::unixtime_t::duration{0U}},
  //                                   n::kInfeasible, mode_id);
  //            }
  //            offsets.emplace_back(iv_at_from_stop.from_, duration, mode_id);
  //            offsets.emplace_back(iv_at_from_stop.to_, n::kInfeasible,
  //            mode_id);
  //          }
  //        }
  //      }
  //    }
  //  }
}

}  // namespace motis::flex