#pragma once

#include <set>

#include "motis/paxmon/graph.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reachability.h"

namespace motis::paxmon {

void update_load(passenger_group* pg, reachability_info const& reachability,
                 passenger_localization const& localization, graph& g);

}  // namespace motis::paxmon
