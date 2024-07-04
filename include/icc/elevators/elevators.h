#pragma once

#include <memory>
#include <string_view>

#include "osr/ways.h"

#include "icc/elevators/match_elevator.h"
#include "icc/elevators/parse_fasta.h"

namespace icc {

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

using elevators_ptr_t = std::shared_ptr<elevators>;

}  // namespace icc