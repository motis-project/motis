#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

extern std::uint64_t initial_over_capacity;

void build_graph_from_journeys(schedule const& sched, paxmon_data& data);

}  // namespace motis::paxmon
