#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/statistics.h"

namespace motis::paxmon::api {

motis::module::msg_ptr get_status(paxmon_data& data,
                                  motis::module::msg_ptr const& msg);

}  // namespace motis::paxmon::api
