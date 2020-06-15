#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/rsl/compact_journey.h"

namespace motis::rsl {

compact_journey to_compact_journey(journey const& j, schedule const& sched);

}  // namespace motis::rsl
