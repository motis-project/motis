#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_trip_capacity(universe& uv, schedule const& sched,
                          capacity_maps const& caps, trip const* trp,
                          bool force_downgrade);

}  // namespace motis::paxmon
