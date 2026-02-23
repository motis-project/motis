#pragma once

#include "motis/elevators/match_elevator.h"
#include "motis/elevators/parse_elevator_id_osm_mapping.h"
#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis {

struct elevators {
  elevators(osr::ways const&,
            elevator_id_osm_mapping_t const*,
            hash_set<osr::node_idx_t> const&,
            vector_map<elevator_idx_t, elevator>&&);

  vector_map<elevator_idx_t, elevator> elevators_;
  point_rtree<elevator_idx_t> elevators_rtree_;
  osr::bitvec<osr::node_idx_t> blocked_;
};

}  // namespace motis