#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

bool check_graph_integrity(universe const& uv, schedule const& sched);

bool check_trip_in_sync(universe const& uv, schedule const& sched,
                        trip const* trp, trip_data_index tdi,
                        bool check_times = true);

bool check_graph_times(universe const& uv, schedule const& sched);

bool check_compact_journey(schedule const& sched, compact_journey const& cj,
                           bool scheduled = false);

bool check_group_routes(universe const& uv);

}  // namespace motis::paxmon
