#include "motis/flex.h"

#include "nigiri/flex.h"

#include "utl/concat.h"

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
                         double const max_matching_distance,
                         std::chrono::seconds const max,
                         nigiri::routing::start_time_t const& start_time,
                         n::routing::td_offsets_t& ret) {
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
      auto [next_stops, first_stop_idx] =
          get_next_stops(tt.flex_transport_stop_seq_[t], lg);
      if (next_stops.empty() || !is_active(t)) {
        continue;
      }
      routings[{lg, next_stops}].emplace_back(t, first_stop_idx);
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
  for (auto const& [area_follow_up_stops, transports] : routings) {
    auto const& [from_stop, follow_up_stops] = area_follow_up_stops;

    auto dest_locations = std::vector<n::location_idx_t>{};
    for (auto const& s : follow_up_stops) {
      s.apply(utl::overloaded{
          [&](n::location_group_idx_t const lg) {
            utl::concat(dest_locations, tt.location_group_locations_[lg]);
          },
          [&](n::flex_area_idx_t const a) {
            utl::concat(dest_locations, tt.flex_area_locations_[a]);
          }});
    }

    auto const destinations =
        utl::to_vec(dest_locations, [&](n::location_idx_t const l) {
          return get_location(r.tt_, r.w_, r.pl_, r.matches_, tt_location{l});
        });

    auto profile = osr::search_profile::kCar;
    auto sharing_data_ptr = static_cast<osr::sharing_data const*>(nullptr);
    from_stop.apply(utl::overloaded{
        [&](n::location_group_idx_t const lg) {
          if (start_allowed.empty()) {
            start_allowed.resize(r.w_->n_nodes());
            end_allowed.resize(r.w_->n_nodes());
            through_allowed.resize(r.w_->n_nodes());
            through_allowed.one_out();
            end_allowed.zero_out();
          }

          start_allowed.zero_out();
          additional_edges.clear();

          auto next_add_node_idx = osr::node_idx_t{r.w_->n_nodes()};
          for (auto const& l : tt.location_group_locations_[lg]) {
            auto const pos =
                get_location(r.tt_, r.w_, r.pl_, r.matches_, tt_location{l});
            auto const l_additional_node_idx = next_add_node_idx++;
            start_allowed.set(l_additional_node_idx, true);

            auto const matches = r.l_->match<osr::foot<false>>(
                pos, false, dir, kMaxGbfsMatchingDistance, nullptr);

            auto& add_node_out = additional_edges[l_additional_node_idx];
            for (auto const& m : matches) {
              auto const handle_node = [&](osr::node_candidate const& node) {
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
                if (utl::find(add_node_out, edge_from_an) ==
                    end(add_node_out)) {
                  add_node_out.emplace_back(edge_from_an);
                }
              };

              handle_node(m.left_);
              handle_node(m.right_);
            }
          }

          profile = osr::search_profile::kCarSharing;
          sharing_data_ptr = &d;
        },
        [](n::flex_area_idx_t) {}});

    auto const paths =
        osr::route(*r.w_, *r.l_, profile, pos, destinations,
                   static_cast<osr::cost_t>(max.count()), dir,
                   max_matching_distance, nullptr, sharing_data_ptr, nullptr);
    for (auto const day_idx : day_idx_iv) {
      for (auto const& [t, stop_idx] : transports) {
        if (!get_traffic_days(t).test(to_idx(day_idx))) {
          continue;
        }

        auto const day =
            tt.internal_interval().from_ + to_idx(day_idx) * date::days{1U};
        auto const stop_time_window =
            tt.flex_transport_stop_time_windows_[t][stop_idx];
        auto const abs_stop_time_iv = n::interval{day + stop_time_window.from_,
                                                  day + stop_time_window.to_};

        for (auto const [p, l] : utl::zip(paths, dest_locations)) {
          if (p.has_value()) {
            auto const duration = n::duration_t{p->cost_ / 60};
            auto const mode_id = static_cast<n::transport_mode_id_t>(profile);

            auto& offsets = ret[l];
            if (offsets.empty()) {
              offsets.emplace_back(n::unixtime_t{n::unixtime_t::duration{0U}},
                                   n::kInfeasible, mode_id);
            }
            offsets.emplace_back(abs_stop_time_iv.from_, duration, mode_id);
            offsets.emplace_back(abs_stop_time_iv.to_, n::kInfeasible, mode_id);
          }
        }
      }
    }
  }
}

}  // namespace motis