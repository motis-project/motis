#pragma once

#include "nigiri/types.h"

#include "osr/types.h"

#include "icc/fwd.h"
#include "icc/point_rtree.h"
#include "icc/types.h"

namespace icc {

point_rtree<elevator_idx_t> create_elevator_rtree(
    nigiri::vector_map<elevator_idx_t, elevator> const&);

osr::hash_set<osr::node_idx_t> get_elevator_nodes(osr::ways const&);

elevator_idx_t match_elevator(
    point_rtree<elevator_idx_t> const&,
    nigiri::vector_map<elevator_idx_t, elevator> const&,
    osr::ways const&,
    osr::node_idx_t);

osr::bitvec<osr::node_idx_t> get_blocked_elevators(
    osr::ways const&,
    nigiri::vector_map<elevator_idx_t, elevator> const&,
    point_rtree<elevator_idx_t> const&,
    osr::hash_set<osr::node_idx_t> const&);

}  // namespace icc