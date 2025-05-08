#include "motis/flex.h"

#include "nigiri/flex.h"

#include "osr/lookup.h"
#include "osr/routing/profiles/foot.h"
#include "osr/ways.h"

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/max_distance.h"

namespace n = nigiri;

namespace motis {

void add_flex_td_offsets(ep::routing const& r,
                         osr::location const& pos,
                         osr::direction const dir,
                         api::PedestrianProfileEnum const,
                         double const max_matching_distance,
                         std::chrono::seconds const max) {
  CISTA_UNUSED_PARAM(max_matching_distance)

  auto const& tt = *r.tt_;
  auto routings =
      hash_map<std::pair<n::flex_stop_t, std::vector<n::flex_stop_t>>,
               std::vector<n::flex_transport_idx_t>>{};

  auto const get_next_stops = [&](auto const& stops, n::flex_stop_t const x) {
    auto next_stops = std::vector<n::flex_stop_t>{};
    auto area_seen = false;
    for (auto c = 0U; c != stops.size(); ++c) {
      auto const stop =
          stops[dir == osr::direction::kForward ? c : stops.size() - c - 1];

      if (area_seen) {
        next_stops.emplace_back(stop);
      }

      if (stop == x) {
        area_seen = true;
      }
    }
    return next_stops;
  };

  // =======================
  // Collect area transports
  // -----------------------
  auto const add_area_flex_transports = [&](n::flex_area_idx_t const a) {
    for (auto const t : tt.flex_area_transports_[a]) {
      auto next_stops = get_next_stops(tt.flex_transport_stop_seq_[t], a);
      if (next_stops.empty()) {
        continue;
      }
      routings[{a, std::move(next_stops)}].emplace_back(t);
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
  r.loc_tree_->find(
      geo::box{pos.pos_, get_max_distance(osr::search_profile::kFoot, max)},
      [&](n::location_idx_t const l) {
        for (auto const lg : tt.location_location_groups_[l]) {
          location_groups.emplace(lg);
        }
        return true;
      });
  for (auto const& lg : location_groups) {
    for (auto const t : tt.location_group_transports_[lg]) {
      auto next_stops = get_next_stops(tt.flex_transport_stop_seq_[t], lg);
      if (next_stops.empty()) {
        continue;
      }
      routings[{lg, next_stops}].emplace_back(t);
    }
  }

  // =======
  // Routing
  // -------
  auto additional_edges =
      osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>{};
  auto start_allowed = osr::bitvec<osr::node_idx_t>{};
  auto end_allowed = osr::bitvec<osr::node_idx_t>{};
  auto through_allowed = osr::bitvec<osr::node_idx_t>{};
  auto d = osr::sharing_data{
      .start_allowed_ = start_allowed,
      .end_allowed_ = end_allowed,
      .through_allowed_ = through_allowed,
      .additional_node_offset_ = r.w_->r_->node_properties_.size(),
      .additional_edges_ = additional_edges};
  CISTA_UNUSED_PARAM(d)
  through_allowed.zero_out();
  for (auto const& [area_follow_up_stops, transports] : routings) {
    auto const& [from_stop, follow_up_stops] = area_follow_up_stops;

    auto next_add_node_idx = osr::node_idx_t{r.w_->n_nodes()};
    start_allowed.zero_out();
    end_allowed.zero_out();
    additional_edges.clear();

    from_stop.apply(utl::overloaded{
        [&](n::location_group_idx_t const lg) {
          for (auto const& l : tt.location_group_locations_[lg]) {
            auto const pos =
                get_location(r.tt_, r.w_, r.pl_, r.matches_, tt_location{l});
            auto const l_additional_node_idx = next_add_node_idx++;
            //            auto& node_add_edges =
            //            additional_edges[l_additional_node_idx];

            auto const matches = r.l_->match<osr::foot<false>>(
                pos, false, dir, kMaxGbfsMatchingDistance, nullptr);

            for (auto const& m : matches) {
              auto const handle_node = [&](osr::node_candidate const node) {
                if (!node.valid() ||
                    node.dist_to_node_ > kMaxGbfsMatchingDistance) {
                  return;
                }

                auto const edge_to_an = osr::additional_edge{
                    l_additional_node_idx,
                    static_cast<osr::distance_t>(node.dist_to_node_)};
                auto& node_edges = additional_edges[node.node_];
                if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
                  node_edges.emplace_back(edge_to_an);
                }

                auto const edge_from_an = osr::additional_edge{
                    node.node_,
                    static_cast<osr::distance_t>(node.dist_to_node_)};
                auto& an_edges = additional_edges[l_additional_node_idx];
                if (utl::find(an_edges, edge_from_an) == end(an_edges)) {
                  an_edges.emplace_back(edge_from_an);
                }
              };

              handle_node(m.left_);
              handle_node(m.right_);
            }
          }
        },
        [](n::flex_area_idx_t const) {

        }});
  }
}

}  // namespace motis