#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

void sync_graph_load(schedule& sched, paxmon_data const& data, trip const* trp);
void sync_graph_load(schedule& sched, paxmon_data const& data);

void verify_graphs_synced(schedule const& sched, paxmon_data const& data);

}  // namespace motis::paxmon
