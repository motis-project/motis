#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void update_trip_formation(schedule const& sched, universe& uv,
                           motis::ris::TripFormationMessage const* tfm);

}  // namespace motis::paxmon
