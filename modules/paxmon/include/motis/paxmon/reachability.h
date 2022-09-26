#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/broken_transfer_info.h"
#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/universe.h"

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

  trip_idx_t trip_idx_{};
  trip_data_index tdi_{};
  journey_leg const leg_{};
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
  std::optional<broken_transfer_info> first_unreachable_transfer_;
};

reachability_info get_reachability(universe const& uv,
                                   fws_compact_journey const& j);

}  // namespace motis::paxmon
