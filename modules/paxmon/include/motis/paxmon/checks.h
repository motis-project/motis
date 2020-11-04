#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/graph.h"

namespace motis::paxmon {

bool check_graph_integrity(graph const& g, schedule const& sched);

bool check_trip_times(graph const& g, schedule const& sched, trip const* trp,
                      trip_data const* td);

bool check_graph_times(graph const& g, schedule const& sched);

}  // namespace motis::paxmon
