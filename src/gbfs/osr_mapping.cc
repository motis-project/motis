#include "motis/gbfs/osr_mapping.h"

#include <optional>
#include <utility>
#include <vector>

#include "tg.h"

#include "geo/box.h"

#include "osr/lookup.h"
#include "osr/routing/profiles/foot.h"
#include "osr/types.h"
#include "osr/ways.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"

#include "motis/constants.h"
#include "motis/types.h"

#include "motis/gbfs/data.h"
#include "motis/gbfs/geofencing.h"

namespace motis::gbfs {

void map_geofencing_zones(gbfs_provider& provider,
                          osr::ways const& w,
                          osr::lookup const& l) {
  auto const make_loc_bitvec = [&]() {
    auto bv = osr::bitvec<osr::node_idx_t>{};
    bv.resize(static_cast<typename osr::bitvec<osr::node_idx_t>::size_type>(
        w.n_nodes() + provider.stations_.size() +
        provider.vehicle_status_.size()));
    return bv;
  };

  auto done = make_loc_bitvec();
  provider.start_allowed_ = make_loc_bitvec();
  provider.end_allowed_ = make_loc_bitvec();
  provider.through_allowed_ = make_loc_bitvec();

  // global rules
  if (!provider.geofencing_zones_.global_rules_.empty()) {
    // vehicle_type_ids currently ignored, using first rule
    auto const& r = provider.geofencing_zones_.global_rules_.front();
    provider.default_restrictions_.ride_start_allowed_ = r.ride_start_allowed_;
    provider.default_restrictions_.ride_end_allowed_ = r.ride_end_allowed_;
    provider.default_restrictions_.ride_through_allowed_ =
        r.ride_through_allowed_;
    provider.default_restrictions_.station_parking_ = r.station_parking_;
  }

  if (provider.default_restrictions_.ride_end_allowed_ &&
      !provider.default_restrictions_.station_parking_) {
    provider.end_allowed_.one_out();
  }
  if (provider.default_restrictions_.ride_through_allowed_) {
    provider.through_allowed_.one_out();
  }
  auto const global_station_parking =
      provider.default_restrictions_.station_parking_.value_or(false);

  auto const handle_point = [&](osr::node_idx_t const n,
                                geo::latlng const& pos) {
    auto start_allowed = std::optional<bool>{};
    auto end_allowed = std::optional<bool>{};
    auto through_allowed = std::optional<bool>{};
    auto station_parking = global_station_parking;
    for (auto const& z : provider.geofencing_zones_.zones_) {
      // check if pos is inside the zone multipolygon
      if (multipoly_contains_point(z.geom_.get(), pos)) {
        // vehicle_type_ids currently ignored, using first rule
        if (!z.rules_.empty()) {
          auto const& r = z.rules_.front();
          start_allowed = r.ride_start_allowed_;
          end_allowed = r.ride_end_allowed_;
          through_allowed = r.ride_through_allowed_;
          if (r.station_parking_.has_value()) {
            station_parking = r.station_parking_.value();
          }
        }
        if (start_allowed.has_value()) {
          break;  // for now
        }
      }
    }
    if (end_allowed.has_value() && !station_parking) {
      provider.end_allowed_.set(n, *end_allowed);
    }
    if (through_allowed.has_value()) {
      provider.through_allowed_.set(n, *through_allowed);
    }
  };

  auto const* osr_r = w.r_.get();
  for (auto const& z : provider.geofencing_zones_.zones_) {
    auto const rect = tg_geom_rect(z.geom_.get());
    auto const bb = geo::box{geo::latlng{rect.min.y, rect.min.x},
                             geo::latlng{rect.max.y, rect.max.x}};

    l.find(bb, [&](osr::way_idx_t const way) {
      for (auto const n : osr_r->way_nodes_[way]) {
        if (done.test(n)) {
          continue;
        }
        done.set(n, true);
        handle_point(n, w.get_node_pos(n).as_latlng());
      }
    });
  }
}

void map_stations(gbfs_provider& provider,
                  osr::ways const& w,
                  osr::lookup const& l) {
  auto next_node_id = static_cast<osr::node_idx_t>(
      w.n_nodes() + provider.additional_nodes_.size());
  for (auto const& [id, st] : provider.stations_) {
    auto const is_renting =
        st.status_.is_renting_ && st.status_.num_vehicles_available_ > 0;
    auto const is_returning = st.status_.is_returning_;

    if (!is_renting && !is_returning) {
      continue;
    }

    auto const matches = l.match<osr::foot<false>>(
        osr::location{st.info_.pos_, osr::level_t{}}, false,
        osr::direction::kForward, kMaxMatchingDistance, nullptr);
    if (matches.empty()) {
      continue;
    }

    auto const additional_node_id = next_node_id++;
    provider.additional_nodes_.emplace_back(
        additional_node{additional_node::station{id}});
    if (is_renting) {
      provider.start_allowed_.set(additional_node_id, true);
    }
    if (is_returning) {
      provider.end_allowed_.set(additional_node_id, true);
      if (st.info_.station_area_) {
        auto const* geom = st.info_.station_area_.get();
        auto const rect = tg_geom_rect(geom);
        auto const bb = geo::box{geo::latlng{rect.min.y, rect.min.x},
                                 geo::latlng{rect.max.y, rect.max.x}};
        auto const* osr_r = w.r_.get();
        l.find(bb, [&](osr::way_idx_t const way) {
          for (auto const n : osr_r->way_nodes_[way]) {
            if (multipoly_contains_point(geom, w.get_node_pos(n).as_latlng())) {
              provider.end_allowed_.set(n, true);
            }
          }
        });
      }
    }

    for (auto const& m : matches) {
      auto const handle_node = [&](osr::node_candidate const node) {
        if (node.valid() && node.dist_to_node_ <= kMaxMatchingDistance) {
          auto const edge_to_an = osr::additional_edge{
              additional_node_id,
              static_cast<osr::distance_t>(node.dist_to_node_)};
          auto& node_edges = provider.additional_edges_[node.node_];
          if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
            node_edges.emplace_back(edge_to_an);
          }

          auto const edge_from_an = osr::additional_edge{
              node.node_, static_cast<osr::distance_t>(node.dist_to_node_)};
          auto& an_edges = provider.additional_edges_[additional_node_id];
          if (utl::find(an_edges, edge_from_an) == end(an_edges)) {
            an_edges.emplace_back(edge_from_an);
          }
        }
      };

      handle_node(m.left_);
      handle_node(m.right_);
    }
  }
}

void map_vehicles(gbfs_provider& provider,
                  osr::ways const& w,
                  osr::lookup const& l) {
  auto next_node_id = static_cast<osr::node_idx_t>(
      w.n_nodes() + provider.additional_nodes_.size());
  for (auto const [vehicle_idx, vs] :
       utl::enumerate(provider.vehicle_status_)) {
    if (vs.is_disabled_ || vs.is_reserved_ || !vs.station_id_.empty() ||
        !vs.home_station_id_.empty()) {
      continue;
    }

    auto const restrictions = provider.geofencing_zones_.get_restrictions(
        vs.pos_, geofencing_restrictions{});
    if (!restrictions.ride_start_allowed_) {
      continue;
    }

    auto const matches = l.match<osr::foot<false>>(
        osr::location{vs.pos_, osr::level_t{}}, false, osr::direction::kForward,
        kMaxMatchingDistance, nullptr);
    if (matches.empty()) {
      continue;
    }

    auto const additional_node_id = next_node_id++;
    provider.additional_nodes_.emplace_back(
        additional_node{additional_node::vehicle{vehicle_idx}});
    provider.start_allowed_.set(additional_node_id, true);

    auto const& add_additional_edges = [&](osr::node_candidate const& nc) {
      auto const edge_to_an = osr::additional_edge{
          additional_node_id, static_cast<osr::distance_t>(nc.dist_to_node_)};
      auto const edge_from_an = osr::additional_edge{
          nc.node_, static_cast<osr::distance_t>(nc.dist_to_node_)};
      auto& node_edges = provider.additional_edges_[nc.node_];
      auto& an_edges = provider.additional_edges_[additional_node_id];

      if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
        node_edges.emplace_back(edge_to_an);
      }
      if (utl::find(an_edges, edge_from_an) == end(an_edges)) {
        an_edges.emplace_back(edge_from_an);
      }
    };

    for (auto const& m : matches) {
      if (m.left_.valid() && m.left_.dist_to_node_ <= kMaxMatchingDistance) {
        add_additional_edges(m.left_);
      }
      if (m.right_.valid() && m.right_.dist_to_node_ <= kMaxMatchingDistance) {
        add_additional_edges(m.right_);
      }
    }
  }
}

}  // namespace motis::gbfs
