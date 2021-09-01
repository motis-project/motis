#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/compact_journey.h"

namespace motis::paxmon {

compact_journey to_compact_journey(journey const& j, schedule const& sched);

}  // namespace motis::paxmon
