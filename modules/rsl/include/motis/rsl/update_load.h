#pragma once

#include <set>

#include "motis/rsl/graph.h"
#include "motis/rsl/localization.h"
#include "motis/rsl/passenger_group.h"
#include "motis/rsl/reachability.h"

namespace motis::rsl {

void update_load(passenger_group* pg, reachability_info const& reachability,
                 passenger_localization const& localization);

}  // namespace motis::rsl
