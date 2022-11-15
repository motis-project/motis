#pragma once

#include <cstdint>

#include "cista/reflection/comparable.h"

namespace motis::paxmon {

using passenger_group_index = std::uint32_t;
using compact_journey_index = std::uint32_t;
using local_group_route_index = std::uint16_t;
using group_route_edges_index = std::uint32_t;
using reroute_log_entry_index = std::uint32_t;

struct passenger_group_with_route {
  CISTA_COMPARABLE()

  passenger_group_index pg_{};
  local_group_route_index route_{};
};

}  // namespace motis::paxmon
