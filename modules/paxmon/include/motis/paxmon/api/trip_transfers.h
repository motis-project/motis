#pragma once

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::api {

motis::module::msg_ptr trip_transfers(paxmon_data& data,
                                      motis::module::msg_ptr const& msg);

}  // namespace motis::paxmon::api
