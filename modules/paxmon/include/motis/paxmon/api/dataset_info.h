#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::api {

motis::module::msg_ptr dataset_info(paxmon_data& data, schedule const& sched);

}  // namespace motis::paxmon::api
