#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/statistics.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_track(schedule const& sched, universe& uv,
                  motis::rt::RtTrackUpdate const* tu,
                  std::vector<edge_index>& updated_interchange_edges);

}  // namespace motis::paxmon
