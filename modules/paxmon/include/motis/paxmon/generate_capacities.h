#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void generate_capacities(schedule const& sched, universe const& uv,
                         std::string const& filename);

}  // namespace motis::paxmon
