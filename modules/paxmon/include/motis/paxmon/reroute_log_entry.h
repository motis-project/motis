#pragma once

#include <cstdint>
#include <iosfwd>

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/broken_transfer_info.h"
#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

enum class reroute_reason_t : std::uint8_t {
  MANUAL,
  BROKEN_TRANSFER,
  MAJOR_DELAY_EXPECTED,
  REVERT_FORECAST,
  SIMULATION,
  UPDATE_FORECAST,
  DESTINATION_UNREACHABLE,
  DESTINATION_REACHABLE
};

struct reroute_log_route_info {
  local_group_route_index route_{};
  float previous_probability_{};
  float new_probability_{};
};

struct reroute_log_localization {
  trip_idx_t trip_idx_{};
  std::uint32_t station_id_{};
  time schedule_arrival_time_{INVALID_TIME};
  time current_arrival_time_{INVALID_TIME};
  bool first_station_{};
  bool in_trip_{};
};

struct reroute_log_entry {
  reroute_log_entry_index index_{};  // for new routes lookup
  reroute_log_route_info old_route_{};
  motis::unixtime system_time_{};  // schedule system timestamp
  motis::unixtime reroute_time_{};  // current time
  reroute_reason_t reason_{reroute_reason_t::MANUAL};
  std::optional<broken_transfer_info> broken_transfer_;
  reroute_log_localization localization_{};
};

inline std::ostream& operator<<(std::ostream& out,
                                reroute_reason_t const reason) {
  switch (reason) {
    case reroute_reason_t::MANUAL: return out << "MANUAL";
    case reroute_reason_t::BROKEN_TRANSFER: return out << "BROKEN_TRANSFER";
    case reroute_reason_t::MAJOR_DELAY_EXPECTED:
      return out << "MAJOR_DELAY_EXPECTED";
    case reroute_reason_t::REVERT_FORECAST: return out << "REVERT_FORECAST";
    case reroute_reason_t::SIMULATION: return out << "SIMULATION";
    case reroute_reason_t::UPDATE_FORECAST: return out << "UPDATE_FORECAST";
    case reroute_reason_t::DESTINATION_UNREACHABLE:
      return out << "DESTINATION_UNREACHABLE";
    case reroute_reason_t::DESTINATION_REACHABLE:
      return out << "DESTINATION_REACHABLE";
  }
  return out;
}

}  // namespace motis::paxmon
