#pragma once

#include "osr/types.h"

#include "motis/elevators/parse_elevator_id_osm_mapping.h"
#include "motis/fwd.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

namespace motis {

point_rtree<elevator_idx_t> create_elevator_rtree(
    vector_map<elevator_idx_t, elevator> const&);

osr::hash_set<osr::node_idx_t> get_elevator_nodes(osr::ways const&);

elevator_idx_t match_elevator(point_rtree<elevator_idx_t> const&,
                              vector_map<elevator_idx_t, elevator> const&,
                              osr::ways const&,
                              osr::node_idx_t);

osr::bitvec<osr::node_idx_t> get_blocked_elevators(
    osr::ways const&,
    elevator_id_osm_mapping_t const*,
    vector_map<elevator_idx_t, elevator> const&,
    point_rtree<elevator_idx_t> const&,
    osr::hash_set<osr::node_idx_t> const&);

}  // namespace motis