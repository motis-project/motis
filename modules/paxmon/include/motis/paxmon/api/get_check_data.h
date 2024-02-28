#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::api {

motis::module::msg_ptr get_check_data(paxmon_data& data, schedule const& sched,
                                      motis::module::msg_ptr const& msg);

motis::module::msg_ptr get_check_data_by_order(
    paxmon_data& data, schedule const& sched,
    motis::module::msg_ptr const& msg);

}  // namespace motis::paxmon::api
