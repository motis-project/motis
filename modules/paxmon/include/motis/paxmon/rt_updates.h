#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/statistics.h"

namespace motis::paxmon {

void handle_rt_update(paxmon_data& data, schedule const& sched,
                      system_statistics& system_stats,
                      tick_statistics& tick_stats,
                      motis::rt::RtUpdates const* update,
                      int arrival_delay_threshold);

std::vector<motis::module::msg_ptr> update_affected_groups(
    paxmon_data& data, schedule const& sched, system_statistics& system_stats,
    tick_statistics& tick_stats, int arrival_delay_threshold,
    int preparation_time);

}  // namespace motis::paxmon
