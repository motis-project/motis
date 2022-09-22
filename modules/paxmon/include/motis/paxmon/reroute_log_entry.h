#pragma once

#include <cstdint>

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

enum class reroute_reason_t : std::uint8_t {
  MANUAL,
  BROKEN_TRANSFER,
  MAJOR_DELAY_EXPECTED,
  REVERT_FORECAST,
  SIMULATION
};

struct reroute_log_entry {
  reroute_log_entry_index index_{};  // for new routes lookup
  local_group_route_index old_route_{};
  float old_route_probability_{};
  motis::unixtime system_time_{};  // schedule system timestamp
  motis::unixtime reroute_time_{};  // current time
  reroute_reason_t reason_{reroute_reason_t::MANUAL};
};

struct reroute_log_new_route {
  local_group_route_index new_route_{};
  float previous_probability_{};
  float new_probability_{};
};

}  // namespace motis::paxmon
