#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void generate_capacities(schedule const& sched, capacity_maps const& caps,
                         universe const& uv, std::string const& filename);

}  // namespace motis::paxmon
