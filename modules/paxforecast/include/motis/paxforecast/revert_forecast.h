#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/index_types.h"
#include "motis/paxmon/universe.h"

namespace motis::paxforecast {

void revert_forecasts(
    motis::paxmon::universe& uv, schedule const& sched,
    std::vector<motis::paxmon::passenger_group_with_route> const& pgwrs);

}  // namespace motis::paxforecast
