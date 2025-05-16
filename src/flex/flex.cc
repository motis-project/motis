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

void for_each_flex_transport(n::timetable const& tt,
                             point_rtree<n::location_idx_t> const& loc_rtree,
                             n::routing::start_time_t const start_time,
                             geo::latlng const& pos,
                             osr::direction const dir,
                             std::chrono::seconds const max,
                             std::function<void(mode_id)> const& fn) {
  // Traffic days helpers.
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

  // Stop index helper.
  auto const get_stop_idx =
      [&](auto const& stops,
          n::flex_stop_t const x) -> std::optional<n::stop_idx_t> {
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
        fn(mode_id{t, *stop_idx, dir});
      }
    }
  };
  tt.flex_area_rtree_.search(pos.lnglat_float(), pos.lnglat_float(),
                             [&](auto&&, auto&&, n::flex_area_idx_t const a) {
                               if (n::is_within(tt, a, pos)) {
                                 add_area_flex_transports(a);
                               }
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
        fn(mode_id{t, *stop_idx, dir});
      }
    }
  }
}

}  // namespace motis::flex