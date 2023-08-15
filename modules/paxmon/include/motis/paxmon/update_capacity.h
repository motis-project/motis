#pragma once

#include <cstdint>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

bool update_trip_capacity(universe& uv, schedule const& sched, trip const* trp,
                          bool track_updates = false);

std::uint32_t update_all_trip_capacities(universe& uv, schedule const& sched,
                                         bool track_updates);

}  // namespace motis::paxmon
