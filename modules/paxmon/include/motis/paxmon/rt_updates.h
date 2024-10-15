#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/statistics.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void handle_rt_update(universe& uv, capacity_maps const& caps,
                      schedule const& sched, motis::rt::RtUpdates const* update,
                      int arrival_delay_threshold);

std::vector<motis::module::msg_ptr> update_affected_groups(
    universe& uv, schedule const& sched, int arrival_delay_threshold,
    int preparation_time);

}  // namespace motis::paxmon
