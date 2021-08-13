#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

enum class reachability_status : std::uint8_t {
  OK,
  BROKEN_INITIAL_ENTRY,
  BROKEN_TRANSFER_EXIT,
  BROKEN_TRANSFER_ENTRY,
  BROKEN_FINAL_EXIT
};

inline std::ostream& operator<<(std::ostream& out, reachability_status t) {
  switch (t) {
    case reachability_status::OK: return out << "OK";
    case reachability_status::BROKEN_INITIAL_ENTRY:
      return out << "BROKEN_INITIAL_ENTRY";
    case reachability_status::BROKEN_TRANSFER_EXIT:
      return out << "BROKEN_TRANSFER_EXIT";
    case reachability_status::BROKEN_TRANSFER_ENTRY:
      return out << "BROKEN_TRANSFER_ENTRY";
    case reachability_status::BROKEN_FINAL_EXIT:
      return out << "BROKEN_FINAL_EXIT";
  }
  return out;
}

struct reachable_trip {
  static constexpr auto const INVALID_INDEX =
      std::numeric_limits<std::size_t>::max();

  trip const* trp_{};
  trip_data* td_{};
  journey_leg const* leg_{};
  time enter_schedule_time_{INVALID_TIME};
  time exit_schedule_time_{INVALID_TIME};
  time enter_real_time_{INVALID_TIME};
  time exit_real_time_{INVALID_TIME};
  std::size_t enter_edge_idx_{INVALID_INDEX};
  std::size_t exit_edge_idx_{INVALID_INDEX};

  bool valid_exit() const { return exit_real_time_ != INVALID_TIME; }
};

struct reachable_station {
  std::uint32_t station_{};
  time schedule_time_{INVALID_TIME};
  time real_time_{INVALID_TIME};
};

struct reachability_info {
  std::vector<reachable_trip> reachable_trips_;
  std::vector<reachable_station> reachable_interchange_stations_;
  bool ok_{true};
  reachability_status status_{reachability_status::OK};
};

reachability_info get_reachability(paxmon_data const& data,
                                   compact_journey const& j);

}  // namespace motis::paxmon
