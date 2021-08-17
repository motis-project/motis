#pragma once

#include <set>

#include "motis/paxmon/localization.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_load(passenger_group* pg, reachability_info const& reachability,
                 passenger_localization const& localization, universe& uv);

}  // namespace motis::paxmon
