#include "motis/elevators/match_elevator.h"

#include "utl/enumerate.h"
#include "utl/parallel_for.h"

#include "osr/ways.h"

namespace motis {

point_rtree<elevator_idx_t> create_elevator_rtree(
    vector_map<elevator_idx_t, elevator> const& elevators) {
  auto t = point_rtree<elevator_idx_t>{};
  for (auto const [i, e] : utl::enumerate(elevators)) {
    t.add(e.pos_, elevator_idx_t{i});
  }
  return t;
}

osr::hash_set<osr::node_idx_t> get_elevator_nodes(osr::ways const& w) {
  auto nodes = osr::hash_set<osr::node_idx_t>{};
  for (auto way = osr::way_idx_t{0U}; way != w.n_ways(); ++way) {
    for (auto const n : w.r_->way_nodes_[way]) {
      if (w.r_->node_properties_[n].is_elevator()) {
        nodes.emplace(n);
      }
    }
  }
  return nodes;
}

elevator_idx_t match_elevator(
    point_rtree<elevator_idx_t> const& rtree,
    vector_map<elevator_idx_t, elevator> const& elevators,
    osr::ways const& w,
    osr::node_idx_t const n) {
  auto const pos = w.get_node_pos(n).as_latlng();
  auto closest = elevator_idx_t::invalid();
  auto closest_dist = std::numeric_limits<double>::max();
  rtree.find(geo::box{pos, 20.0}, [&](elevator_idx_t const e) {
    auto const dist = geo::distance(elevators[e].pos_, pos);
    if (dist < 20 && dist < closest_dist) {
      closest_dist = dist;
      closest = e;
    }
  });
  return closest;
}

osr::bitvec<osr::node_idx_t> get_blocked_elevators(
    osr::ways const& w,
    elevator_id_osm_mapping_t const* ids,
    vector_map<elevator_idx_t, elevator> const& elevators,
    point_rtree<elevator_idx_t> const& elevators_rtree,
    osr::hash_set<osr::node_idx_t> const& elevator_nodes) {
  auto inactive = osr::hash_set<osr::node_idx_t>{};
  auto inactive_mutex = std::mutex{};

  auto id_to_elevator = hash_map<std::string, elevator_idx_t>{};
  for (auto const [i, e] : utl::enumerate(elevators)) {
    if (e.id_str_.has_value()) {
      id_to_elevator.emplace(*e.id_str_, elevator_idx_t{i});
    }
  }

  utl::parallel_for(elevator_nodes, [&](osr::node_idx_t const n) {
    auto e = elevator_idx_t::invalid();

    // SIRI-FM matching by DIID:
    // node idx -> OSM ID -> DIID -> elevator_idx
    if (ids != nullptr) {
      auto const id_it = ids->find(to_idx(w.node_to_osm_[n]));
      if (id_it != end(*ids)) {
        auto const e_it = id_to_elevator.find(id_it->second);
        if (e_it != end(id_to_elevator)) {
          e = e_it->second;
        }
      }
    }

    // DB FaSta API: geomatching
    if (e == elevator_idx_t::invalid()) {
      e = match_elevator(elevators_rtree, elevators, w, n);
    }

    if (e != elevator_idx_t::invalid() && !elevators[e].status_) {
      auto const lock = std::scoped_lock{inactive_mutex};
      inactive.emplace(n);
    }
  });
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  blocked.resize(w.n_nodes());
  for (auto const n : inactive) {
    blocked.set(n, true);
  }
  return blocked;
}

}  // namespace motis
