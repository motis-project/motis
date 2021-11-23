#pragma once

#include <tuple>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/reachability.h"

namespace motis::paxmon {

struct passenger_localization {
  trip const* in_trip_{};
  station const* at_station_{};
  time schedule_arrival_time_{INVALID_TIME};
  time current_arrival_time_{INVALID_TIME};
  bool first_station_{false};

  inline friend bool operator==(passenger_localization const& lhs,
                                passenger_localization const& rhs) {
    return std::tie(lhs.in_trip_, lhs.at_station_, lhs.current_arrival_time_,
                    lhs.first_station_) ==
           std::tie(rhs.in_trip_, rhs.at_station_, rhs.current_arrival_time_,
                    rhs.first_station_);
  }

  inline friend bool operator!=(passenger_localization const& lhs,
                                passenger_localization const& rhs) {
    return std::tie(lhs.in_trip_, lhs.at_station_, lhs.current_arrival_time_,
                    lhs.first_station_) !=
           std::tie(rhs.in_trip_, rhs.at_station_, rhs.current_arrival_time_,
                    rhs.first_station_);
  }

  bool in_trip() const { return in_trip_ != nullptr; }
};

passenger_localization localize(schedule const& sched,
                                reachability_info const& reachability,
                                time localization_time);

}  // namespace motis::paxmon
