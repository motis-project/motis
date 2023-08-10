#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/trip_capacity_status.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_trip_capacity_status(schedule const& sched, universe& uv,
                                 trip const* trp, trip_data_index tdi);

}  // namespace motis::paxmon
