#pragma once

#include <vector>

#include "osr/routing/additional_edge.h"
#include "osr/routing/sharing_data.h"
#include "osr/types.h"
#include "osr/ways.h"

#include "nigiri/types.h"

namespace motis::flex {

struct flex_routing_data {
  osr::sharing_data to_sharing_data() {
    return {.start_allowed_ = &start_allowed_,
            .end_allowed_ = &end_allowed_,
            .through_allowed_ = &through_allowed_,
            .additional_node_offset_ = additional_node_offset_,
            .additional_node_coordinates_ = additional_node_coordinates_,
            .additional_edges_ = additional_edges_};
  }

  nigiri::location_idx_t get_additional_node(osr::node_idx_t const n) const {
    return additional_nodes_[to_idx(n - additional_node_offset_)];
  }

  osr::bitvec<osr::node_idx_t> start_allowed_;
  osr::bitvec<osr::node_idx_t> through_allowed_;
  osr::bitvec<osr::node_idx_t> end_allowed_;
  osr::node_idx_t::value_t additional_node_offset_;
  std::vector<geo::latlng> additional_node_coordinates_;
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>
      additional_edges_;
  std::vector<nigiri::location_idx_t> additional_nodes_;
};

}  // namespace motis::flex
