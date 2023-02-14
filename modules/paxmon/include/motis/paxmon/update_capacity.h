#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_trip_capacity(universe& uv, schedule const& sched, trip const* trp);

void update_all_trip_capacities(universe& uv, schedule const& sched);

}  // namespace motis::paxmon
