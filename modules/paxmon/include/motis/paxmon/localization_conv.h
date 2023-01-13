#pragma once

#include "motis/paxmon/localization.h"
#include "motis/paxmon/reroute_log_entry.h"

namespace motis::paxmon {

inline reroute_log_localization to_log_localization(
    passenger_localization const& loc) {
  return {loc.in_trip() ? loc.in_trip_->trip_idx_ : 0,
          loc.at_station_->index_,
          loc.schedule_arrival_time_,
          loc.current_arrival_time_,
          loc.first_station_,
          loc.in_trip()};
}

}  // namespace motis::paxmon
