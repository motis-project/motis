#pragma once

#include <cstdint>
#include <limits>

#include "cista/reflection/comparable.h"

namespace motis::paxmon {

using event_node_index = std::uint32_t;
auto const constexpr INVALID_EVENT_NODE_INDEX =
    std::numeric_limits<event_node_index>::max();

struct universe;
struct edge;

struct edge_index {
  CISTA_COMPARABLE()

  edge* get(universe const&) const;

  event_node_index node_{};
  std::uint32_t out_edge_idx_{};
};

edge_index get_edge_index(universe const&, edge const*);

using trip_data_index = std::uint32_t;
auto const constexpr INVALID_TRIP_DATA_INDEX =
    std::numeric_limits<trip_data_index>::max();

}  // namespace motis::paxmon
