#pragma once

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxforecast {

struct paxforecast;

motis::module::msg_ptr apply_measures(paxforecast& mod,
                                      motis::paxmon::paxmon_data& data,
                                      motis::module::msg_ptr const& msg);

}  // namespace motis::paxforecast
