#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/rt_update_context.h"

namespace motis::paxmon::api {

motis::module::msg_ptr add_groups(schedule const& sched, paxmon_data& data,
                                  rt_update_context& rt_update_ctx,
                                  bool allow_reuse,
                                  motis::module::msg_ptr const& msg);

}  // namespace motis::paxmon::api
