#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/universe.h"

#include "motis/paxforecast/load_forecast.h"
#include "motis/paxforecast/simulation_result.h"

namespace motis::paxforecast {

motis::module::msg_ptr make_forecast_update_msg(
    schedule const& sched, motis::paxmon::universe const& uv,
    simulation_result const& sim_result, load_forecast const& lfc);

}  // namespace motis::paxforecast
