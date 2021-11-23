#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void write_over_capacity_report(universe const& uv, schedule const& sched,
                                std::string const& filename);

}  // namespace motis::paxmon
