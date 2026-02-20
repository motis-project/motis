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
#include "utl/to_vec.h"
#include "utl/zip.h"

#include "motis/constants.h"
#include "motis/types.h"

#include "motis/box_rtree.h"
#include "motis/gbfs/compression.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/geofencing.h"

namespace motis::gbfs {

struct node_match {
  osr::way_candidate const& way() const { return wc_; }
  osr::node_candidate const& node() const {
    return left_ ? wc_.left_ : wc_.right_;
  }

  osr::way_candidate wc_;
  bool left_{};
};

struct osr_mapping {
  osr_mapping(osr::ways const& w,
              osr::lookup const& l,
              gbfs_provider const& provider)
      : w_{w}, l_{l}, provider_{provider} {
    products_data_.resize(provider.products_.size());
  }

  void map_geofencing_zones() {
    auto const make_loc_bitvec = [&]() {
      auto bv = osr::bitvec<osr::node_idx_t>{};
      bv.resize(static_cast<typename osr::bitvec<osr::node_idx_t>::size_type>(
          w_.n_nodes() + provider_.stations_.size() +
          provider_.vehicle_status_.size()));
      return bv;
    };

    auto zone_rtree = box_rtree<std::size_t>{};
    for (auto const [i, z] :
         utl::enumerate(provider_.geofencing_zones_.zones_)) {
      zone_rtree.add(z.bounding_box(), i);
    }

    for (auto [prod, rd] : utl::zip(provider_.products_, products_data_)) {
      auto default_restrictions = provider_.default_restrictions_;
      rd.start_allowed_ = make_loc_bitvec();
      rd.end_allowed_ = make_loc_bitvec();
      rd.through_allowed_ = make_loc_bitvec();

      // global rules
      for (auto const& r : provider_.geofencing_zones_.global_rules_) {
        if (!applies(r.vehicle_type_idxs_, prod.vehicle_types_)) {
          continue;
        }
        default_restrictions.ride_start_allowed_ = r.ride_start_allowed_;
        default_restrictions.ride_end_allowed_ = r.ride_end_allowed_;
        default_restrictions.ride_through_allowed_ = r.ride_through_allowed_;
        default_restrictions.station_parking_ = r.station_parking_;
        break;
      }

      if ((prod.return_constraint_ == return_constraint::kAnyStation ||
           prod.return_constraint_ == return_constraint::kRoundtripStation) &&
          (prod.known_return_constraint_ ||
           provider_.geofencing_zones_.zones_.empty()) &&
          !default_restrictions.station_parking_.has_value()) {
        default_restrictions.station_parking_ = true;
      }

      if (default_restrictions.ride_end_allowed_ &&
          !default_restrictions.station_parking_.value_or(false)) {
        rd.end_allowed_.one_out();
      }
      if (default_restrictions.ride_through_allowed_) {
        rd.through_allowed_.one_out();
      }

      rd.station_parking_ =
          default_restrictions.station_parking_.value_or(false);
    }

    auto done = make_loc_bitvec();

    auto zone_indices = std::vector<std::size_t>{};
    zone_indices.reserve(provider_.geofencing_zones_.zones_.size());
    auto const handle_point = [&](osr::node_idx_t const n,
                                  geo::latlng const& pos) {
      for (auto [prod, rd] : utl::zip(provider_.products_, products_data_)) {
        auto start_allowed = std::optional<bool>{};
        auto end_allowed = std::optional<bool>{};
        auto through_allowed = std::optional<bool>{};
        auto station_parking = rd.station_parking_;

        // zones have to be checked in the order they are defined
        zone_indices.clear();
        zone_rtree.find(pos, [&](std::size_t const zone_idx) {
          zone_indices.push_back(zone_idx);
        });
        utl::sort(zone_indices);

        for (auto const zone_idx : zone_indices) {
          auto const& z = provider_.geofencing_zones_.zones_[zone_idx];
          // check if pos is inside the zone multipolygon
          if (multipoly_contains_point(z.geom_.get(), pos)) {
            for (auto const& r : z.rules_) {
              if (!applies(r.vehicle_type_idxs_, prod.vehicle_types_)) {
                continue;
              }
              if (r.station_parking_.has_value()) {
                station_parking = r.station_parking_.value();
              }
              start_allowed = r.ride_start_allowed_;
              end_allowed = r.ride_end_allowed_ && !station_parking;
              through_allowed = r.ride_through_allowed_;
              break;
            }
            if (start_allowed.has_value()) {
              break;  // for now
            }
          }
        }
        if (end_allowed.has_value()) {
          rd.end_allowed_.set(n, *end_allowed);
        }
        if (through_allowed.has_value()) {
          rd.through_allowed_.set(n, *through_allowed);
        }
      }
    };

    auto const* osr_r = w_.r_.get();
    for (auto const& z : provider_.geofencing_zones_.zones_) {
      l_.find(z.bounding_box(), [&](osr::way_idx_t const way) {
        for (auto const n : osr_r->way_nodes_[way]) {
          if (done.test(n)) {
            continue;
          }
          done.set(n, true);
          handle_point(n, w_.get_node_pos(n).as_latlng());
        }
      });
    }
  }

  std::vector<node_match> get_node_matches(osr::location const& loc) const {
    using footp = osr::bike_sharing::footp;
    using bikep = osr::bike_sharing::bikep;
    static constexpr auto foot_params = osr::bike_sharing::footp::parameters{};
    static constexpr auto bike_params = osr::bike_sharing::bikep::parameters{};

    auto is_acceptable_node = [&](osr::node_candidate const& n) {
      if (!n.valid() || n.dist_to_node_ > kMaxGbfsMatchingDistance) {
        return false;
      }
      auto const& node_props = w_.r_->node_properties_[n.node_];
      if (footp::node_cost(foot_params, node_props) == osr::kInfeasible ||
          bikep::node_cost(bike_params, node_props) == osr::kInfeasible) {
        return false;
      }
      // node needs to have at least one way accessible by foot and one by bike
      return utl::any_of(w_.r_->node_ways_[n.node_],
                         [&](auto const way_idx) {
                           return footp::way_cost(
                                      footp::parameters{},
                                      w_.r_->way_properties_[way_idx],
                                      osr::direction::kForward,
                                      0U) != osr::kInfeasible;
                         }) &&
             utl::any_of(w_.r_->node_ways_[n.node_], [&](auto const way_idx) {
               return bikep::way_cost(
                          bikep::parameters{}, w_.r_->way_properties_[way_idx],
                          osr::direction::kForward, 0U) != osr::kInfeasible;
             });
    };

    auto const matches = l_.match<footp>(footp::parameters{}, loc, false,
                                         osr::direction::kForward,
                                         kMaxGbfsMatchingDistance, nullptr);
    auto node_matches = std::vector<node_match>{};
    for (auto const& m : matches) {
      if (is_acceptable_node(m.left_)) {
        node_matches.emplace_back(node_match{m, true});
      }
      if (is_acceptable_node(m.right_)) {
        node_matches.emplace_back(node_match{m, false});
      }
    }
    utl::sort(node_matches, [](auto const& a, auto const& b) {
      return a.node().dist_to_node_ < b.node().dist_to_node_;
    });

    auto connected_components = hash_set<osr::component_idx_t>{};
    for (auto it = node_matches.begin(); it != node_matches.end();) {
      auto const component = w_.r_->way_component_[it->way().way_];
      if (!connected_components.insert(component).second) {
        it = node_matches.erase(it);
      } else {
        ++it;
      }
    }

    return node_matches;
  }

  void map_stations() {
    for (auto [prod_b, rd_b] : utl::zip(provider_.products_, products_data_)) {
      auto& prod = prod_b;  // fix for apple clang
      auto& rd = rd_b;
      for (auto const& [id, st] : provider_.stations_) {
        auto is_renting =
            st.status_.is_renting_ && st.status_.num_vehicles_available_ > 0;
        auto is_returning = st.status_.is_returning_;

        // if the station lists vehicles available by type, at least one of
        // the vehicle types included in the product segment must be available
        if (is_renting && !st.status_.vehicle_types_available_.empty()) {
          is_renting = utl::any_of(
              st.status_.vehicle_types_available_, [&](auto const& vt) {
                return vt.second != 0 && prod.includes_vehicle_type(vt.first);
              });
        }

        // same for returning vehicles
        if (is_returning && !st.status_.vehicle_docks_available_.empty()) {
          is_returning = utl::any_of(
              st.status_.vehicle_docks_available_, [&](auto const& vt) {
                return vt.second != 0 && prod.includes_vehicle_type(vt.first);
              });
        }

        if (!is_renting && !is_returning) {
          continue;
        }

        auto const matches =
            get_node_matches(osr::location{st.info_.pos_, osr::level_t{}});
        if (matches.empty()) {
          continue;
        }

        auto const additional_node_id = add_node(
            rd, additional_node{additional_node::station{id}}, st.info_.pos_);
        if (is_renting) {
          rd.start_allowed_.set(additional_node_id, true);
        }
        if (is_returning) {
          rd.end_allowed_.set(additional_node_id, true);
          if (st.info_.station_area_) {
            auto const* geom = st.info_.station_area_.get();
            auto const rect = tg_geom_rect(geom);
            auto const bb = geo::box{geo::latlng{rect.min.y, rect.min.x},
                                     geo::latlng{rect.max.y, rect.max.x}};
            auto const* osr_r = w_.r_.get();
            l_.find(bb, [&](osr::way_idx_t const way) {
              for (auto const n : osr_r->way_nodes_[way]) {
                if (multipoly_contains_point(geom,
                                             w_.get_node_pos(n).as_latlng())) {
                  rd.end_allowed_.set(n, true);
                }
              }
            });
          }
        }
        for (auto const& m : matches) {
          auto const& node = m.node();
          auto const edge_to_an = osr::additional_edge{
              additional_node_id,
              static_cast<osr::distance_t>(node.dist_to_node_)};
          auto& node_edges = rd.additional_edges_[node.node_];
          if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
            node_edges.emplace_back(edge_to_an);
          }

          auto const edge_from_an = osr::additional_edge{
              node.node_, static_cast<osr::distance_t>(node.dist_to_node_)};
          auto& an_edges = rd.additional_edges_[additional_node_id];
          if (utl::find(an_edges, edge_from_an) == end(an_edges)) {
            an_edges.emplace_back(edge_from_an);
          }
        }
      }
    }
  }

  void map_vehicles() {
    for (auto [prod, rd] : utl::zip(provider_.products_, products_data_)) {
      for (auto const [vehicle_idx, vs] :
           utl::enumerate(provider_.vehicle_status_)) {
        if (vs.is_disabled_ || vs.is_reserved_ ||
            !prod.includes_vehicle_type(vs.vehicle_type_idx_)) {
          continue;
        }

        auto const restrictions = provider_.geofencing_zones_.get_restrictions(
            vs.pos_, vs.vehicle_type_idx_, geofencing_restrictions{});
        if (!restrictions.ride_start_allowed_) {
          continue;
        }

        auto const matches =
            get_node_matches(osr::location{vs.pos_, osr::level_t{}});
        if (matches.empty()) {
          continue;
        }

        auto const additional_node_id =
            add_node(rd, additional_node{additional_node::vehicle{vehicle_idx}},
                     vs.pos_);
        rd.start_allowed_.set(additional_node_id, true);

        for (auto const& m : matches) {
          auto const& nc = m.node();
          auto const edge_to_an = osr::additional_edge{
              additional_node_id,
              static_cast<osr::distance_t>(nc.dist_to_node_)};
          auto& node_edges = rd.additional_edges_[nc.node_];
          if (utl::find(node_edges, edge_to_an) == end(node_edges)) {
            node_edges.push_back(edge_to_an);
          }

          auto const edge_from_an = osr::additional_edge{
              nc.node_, static_cast<osr::distance_t>(nc.dist_to_node_)};
          auto& an_edges = rd.additional_edges_[additional_node_id];
          if (utl::find(an_edges, edge_from_an) == end(an_edges)) {
            an_edges.push_back(edge_from_an);
          }
        }
      }
    }
  }

  osr::node_idx_t add_node(routing_data& rd,
                           additional_node&& an,
                           geo::latlng const& pos) const {
    auto const node_id = static_cast<osr::node_idx_t>(
        w_.n_nodes() + rd.additional_nodes_.size());
    rd.additional_nodes_.push_back(std::move(an));
    rd.additional_node_coordinates_.push_back(pos);
    assert(rd.start_allowed_.size() >=
           static_cast<typename osr::bitvec<osr::node_idx_t>::size_type>(
               node_id + 1));
    return node_id;
  }

  osr::ways const& w_;
  osr::lookup const& l_;
  gbfs_provider const& provider_;

  std::vector<routing_data> products_data_;
};

void map_data(osr::ways const& w,
              osr::lookup const& l,
              gbfs_provider const& provider,
              provider_routing_data& prd) {
  auto mapping = osr_mapping{w, l, provider};
  mapping.map_geofencing_zones();
  mapping.map_stations();
  mapping.map_vehicles();

  prd.products_ = utl::to_vec(mapping.products_data_, [&](routing_data& rd) {
    return compressed_routing_data{
        .additional_nodes_ = std::move(rd.additional_nodes_),
        .additional_node_coordinates_ =
            std::move(rd.additional_node_coordinates_),
        .additional_edges_ = std::move(rd.additional_edges_),
        .start_allowed_ = compress_bitvec(rd.start_allowed_),
        .end_allowed_ = compress_bitvec(rd.end_allowed_),
        .through_allowed_ = compress_bitvec(rd.through_allowed_)};
  });
}

}  // namespace motis::gbfs
