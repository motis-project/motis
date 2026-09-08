#pragma once

#include <vector>

#include "osr/routing/additional_edge.h"
#include "osr/routing/sharing_data.h"
#include "osr/types.h"
#include "osr/ways.h"

#include "nigiri/types.h"

namespace motis::flex {

// The additional nodes of one flex routing: the timetable location each of
// them stands for, plus the coordinates and edges the street search needs.
//
// Built per routing group. A search that is kept alive to reconstruct legs
// later is handed these along with it, because the next group rebuilds them.
struct flex_additional_nodes {
  void reset(osr::node_idx_t::value_t const offset) {
    offset_ = offset;
    locations_.clear();
    coordinates_.clear();
    edges_.clear();
  }

  nigiri::location_idx_t get(osr::node_idx_t const n) const {
    return locations_[to_idx(n - offset_)];
  }

  // Without the `*_allowed_` bitvecs: reconstructing an already-found path
  // does not re-check the access constraints, and keeping a copy of them per
  // routing group (3 x n_nodes bits) would be prohibitive.
  osr::sharing_data to_sharing_data() const {
    return {.additional_node_offset_ = offset_,
            .additional_node_coordinates_ = coordinates_,
            .additional_edges_ = edges_};
  }

  osr::node_idx_t::value_t offset_{};
  std::vector<nigiri::location_idx_t> locations_;
  std::vector<geo::latlng> coordinates_;
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>> edges_;
};

// Scratch space of one flex routing. The bitvecs are reused across routing
// groups; `additional_nodes_` is refilled by each `prepare_sharing_data()`
// call, and moved out by callers that need it to outlive the routing.
struct flex_routing_data {
  osr::sharing_data to_sharing_data() const {
    auto s = additional_nodes_.to_sharing_data();
    s.start_allowed_ = &start_allowed_;
    s.end_allowed_ = &end_allowed_;
    s.through_allowed_ = &through_allowed_;
    return s;
  }

  nigiri::location_idx_t get_additional_node(osr::node_idx_t const n) const {
    return additional_nodes_.get(n);
  }

  osr::bitvec<osr::node_idx_t> start_allowed_;
  osr::bitvec<osr::node_idx_t> through_allowed_;
  osr::bitvec<osr::node_idx_t> end_allowed_;
  flex_additional_nodes additional_nodes_;
};

}  // namespace motis::flex
