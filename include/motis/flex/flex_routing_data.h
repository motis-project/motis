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

// Additional node data of one flex routing without the allowed bitvecs (3 x
// n_nodes bits, ~40 ms to build on the planet): all that reconstructing paths
// from a retained search and annotating the resulting legs need.
struct retained_flex_data {
  explicit retained_flex_data(flex_routing_data const& frd)
      : sharing_{.start_allowed_ = nullptr,
                 .end_allowed_ = nullptr,
                 .through_allowed_ = nullptr,
                 .additional_node_offset_ = frd.additional_node_offset_,
                 .additional_node_coordinates_ = frd_.additional_node_coordinates_,
                 .additional_edges_ = frd_.additional_edges_} {
    frd_.additional_node_offset_ = frd.additional_node_offset_;
    frd_.additional_node_coordinates_ = frd.additional_node_coordinates_;
    frd_.additional_edges_ = frd.additional_edges_;
    frd_.additional_nodes_ = frd.additional_nodes_;
  }
  retained_flex_data(retained_flex_data const&) = delete;
  retained_flex_data& operator=(retained_flex_data const&) = delete;

  flex_routing_data frd_;  // bitvecs stay empty
  osr::sharing_data sharing_;  // references frd_
};

}  // namespace motis::flex
