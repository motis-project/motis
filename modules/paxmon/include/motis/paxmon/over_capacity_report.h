#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

void write_over_capacity_report(paxmon_data const& data, schedule const& sched,
                                std::string const& filename);

}  // namespace motis::paxmon
