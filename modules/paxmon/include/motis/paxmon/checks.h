#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/graph.h"

namespace motis::paxmon {

bool check_graph_integrity(graph const& g, schedule const& sched);

bool check_graph_times(graph const& g, schedule const& sched);

bool check_compact_journey(schedule const& sched, compact_journey const& cj,
                           bool scheduled = false);

}  // namespace motis::paxmon
