#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

namespace motis::revise {

void update_journey_status(schedule const& sched, journey&);

}  // namespace motis::revise
