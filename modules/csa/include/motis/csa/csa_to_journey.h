#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/csa/csa_journey.h"

namespace motis::csa {

journey csa_to_journey(schedule const& sched, csa_journey const& csa);

}  // namespace motis::csa
