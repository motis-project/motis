#pragma once

#include "nigiri/types.h"

#include "icc/parse_fasta.h"
#include "icc/point_rtree.h"

namespace icc {

elevator_idx_t match_elevator(
    point_rtree<elevator_idx_t> const& rtree,
    nigiri::vector_map<elevator_idx_t, elevator> const& elevators,
    osr::ways const& w,
    osr::node_idx_t const n) {
  auto const pos = w.get_node_pos(n).as_latlng();
  auto closest = elevator_idx_t::invalid();
  auto closest_dist = std::numeric_limits<double>::max();
  rtree.find(pos, [&](elevator_idx_t const e) {
    auto const dist = geo::distance(elevators[e].pos_, pos);
    if (dist < 20 && dist < closest_dist) {
      closest_dist = dist;
      closest = e;
    }
  });
  return closest;
}

}  // namespace icc