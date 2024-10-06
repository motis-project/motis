#pragma once

#include "motis/elevators/match_elevator.h"
#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis {

struct elevators {
  elevators(osr::ways const& w,
            hash_set<osr::node_idx_t> const& elevator_nodes,
            vector_map<elevator_idx_t, elevator>&& elevators)
      : elevators_{std::move(elevators)},
        elevators_rtree_{create_elevator_rtree(elevators_)},
        blocked_{get_blocked_elevators(
            w, elevators_, elevators_rtree_, elevator_nodes)} {}

  vector_map<elevator_idx_t, elevator> elevators_;
  point_rtree<elevator_idx_t> elevators_rtree_;
  osr::bitvec<osr::node_idx_t> blocked_;
};

}  // namespace motis