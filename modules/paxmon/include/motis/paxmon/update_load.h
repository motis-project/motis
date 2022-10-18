#pragma once

#include <set>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/index_types.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_load(passenger_group_with_route pgwr,
                 reachability_info const& reachability,
                 passenger_localization const& localization, universe& uv,
                 schedule const& sched);

}  // namespace motis::paxmon
