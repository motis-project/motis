#pragma once

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/protocol/PaxMonFilterTripsTimeFilter_generated.h"

namespace motis::paxmon::api {

inline bool include_trip_based_on_time_filter(
    trip const* trp, PaxMonFilterTripsTimeFilter const filter,
    time const filter_interval_begin, time const filter_interval_end) {
  auto const dep = trp->id_.primary_.get_time();
  auto const arr = trp->id_.secondary_.target_time_;
  if (filter == PaxMonFilterTripsTimeFilter_DepartureTime) {
    if (dep < filter_interval_begin || dep >= filter_interval_end) {
      return false;
    }
  } else if (filter == PaxMonFilterTripsTimeFilter_DepartureOrArrivalTime) {
    if ((dep < filter_interval_begin || dep >= filter_interval_end) &&
        (arr < filter_interval_begin || arr >= filter_interval_end)) {
      return false;
    }
  } else if (filter == PaxMonFilterTripsTimeFilter_ActiveTime) {
    if (dep > filter_interval_end || arr < filter_interval_begin) {
      return false;
    }
  }
  return true;
}

}  // namespace motis::paxmon::api
