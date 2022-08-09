#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/compact_journey.h"

namespace motis::paxmon {

compact_journey to_compact_journey(journey const& j, schedule const& sched);

fws_compact_journey to_fws_compact_journey(
    journey const& j, schedule const& sched,
    dynamic_fws_multimap<journey_leg>& fws);

}  // namespace motis::paxmon
