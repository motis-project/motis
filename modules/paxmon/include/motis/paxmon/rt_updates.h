#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/rt_update_context.h"
#include "motis/paxmon/statistics.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void handle_rt_update(universe& uv, capacity_maps const& caps,
                      schedule const& sched, rt_update_context& rt_ctx,
                      system_statistics& system_stats,
                      tick_statistics& tick_stats,
                      motis::rt::RtUpdates const* update,
                      int arrival_delay_threshold);

std::vector<motis::module::msg_ptr> update_affected_groups(
    universe& uv, schedule const& sched, rt_update_context& rt_ctx,
    system_statistics& system_stats, tick_statistics& tick_stats,
    int arrival_delay_threshold, int preparation_time);

}  // namespace motis::paxmon
