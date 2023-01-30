#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/index_types.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/universe.h"

#include "motis/paxforecast/simulation_result.h"

namespace motis::paxforecast {

void revert_forecasts(
    motis::paxmon::universe& uv, schedule const& sched,
    simulation_result const& sim_result,
    std::vector<motis::paxmon::passenger_group_with_route> const& pgwrs,
    mcd::hash_map<motis::paxmon::passenger_group_with_route,
                  motis::paxmon::passenger_localization const*> const&
        pgwr_localizations);

}  // namespace motis::paxforecast
