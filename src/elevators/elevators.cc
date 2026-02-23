#include "motis/elevators/elevators.h"

#include "osr/ways.h"

namespace motis {

vector_map<elevator_idx_t, elevator> update_elevator_coordinates(
    osr::ways const& w,
    elevator_id_osm_mapping_t const* ids,
    hash_set<osr::node_idx_t> const& elevator_nodes,
    vector_map<elevator_idx_t, elevator>&& elevators) {
  if (ids != nullptr) {
    auto id_to_elevator = hash_map<std::string, elevator_idx_t>{};
    for (auto const [i, e] : utl::enumerate(elevators)) {
      if (e.id_str_.has_value()) {
        id_to_elevator.emplace(*e.id_str_, elevator_idx_t{i});
      }
    }

    for (auto const n : elevator_nodes) {
      auto const id_it = ids->find(cista::to_idx(w.node_to_osm_[n]));
      if (id_it != end(*ids)) {
        auto const e_it = id_to_elevator.find(id_it->second);
        if (e_it != end(id_to_elevator)) {
          elevators[e_it->second].pos_ = w.get_node_pos(n).as_latlng();
        }
      }
    }
  }
  return elevators;
}

elevators::elevators(osr::ways const& w,
                     elevator_id_osm_mapping_t const* ids,
                     hash_set<osr::node_idx_t> const& elevator_nodes,
                     vector_map<elevator_idx_t, elevator>&& elevators)
    : elevators_{update_elevator_coordinates(
          w, ids, elevator_nodes, std::move(elevators))},
      elevators_rtree_{create_elevator_rtree(elevators_)},
      blocked_{get_blocked_elevators(
          w, ids, elevators_, elevators_rtree_, elevator_nodes)} {}

}  // namespace motis