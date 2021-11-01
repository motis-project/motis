#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/rt_update_context.h"

namespace motis::paxmon::api {

motis::module::msg_ptr remove_groups(schedule const& sched, paxmon_data& data,
                                     rt_update_context& rt_update_ctx,
                                     bool keep_group_history,
                                     bool check_graph_integrity_end,
                                     motis::module::msg_ptr const& msg);

}  // namespace motis::paxmon::api
