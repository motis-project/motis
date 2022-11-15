#pragma once

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::api {

motis::module::msg_ptr remove_groups(paxmon_data& data,
                                     bool check_graph_integrity_end,
                                     motis::module::msg_ptr const& msg);

}  // namespace motis::paxmon::api
