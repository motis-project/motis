#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/localization.h"

namespace motis::paxmon {

compact_journey get_prefix(schedule const& sched, compact_journey const& cj,
                           passenger_localization const& loc);

compact_journey merge_journeys(schedule const& sched,
                               compact_journey const& prefix,
                               compact_journey const& suffix);

}  // namespace motis::paxmon
