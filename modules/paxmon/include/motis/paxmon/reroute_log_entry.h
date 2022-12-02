#pragma once

#include <cstdint>

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/broken_transfer_info.h"
#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

enum class reroute_reason_t : std::uint8_t {
  MANUAL,
  BROKEN_TRANSFER,
  MAJOR_DELAY_EXPECTED,
  REVERT_FORECAST,
  SIMULATION,
  UPDATE_FORECAST
};

struct reroute_log_route_info {
  local_group_route_index route_{};
  float previous_probability_{};
  float new_probability_{};
};

struct reroute_log_entry {
  reroute_log_entry_index index_{};  // for new routes lookup
  reroute_log_route_info old_route_{};
  motis::unixtime system_time_{};  // schedule system timestamp
  motis::unixtime reroute_time_{};  // current time
  reroute_reason_t reason_{reroute_reason_t::MANUAL};
  std::optional<broken_transfer_info> broken_transfer_;
};

}  // namespace motis::paxmon
