#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"
#include "motis/tripbased/data.h"
#include "motis/tripbased/tb_journey.h"

namespace motis::tripbased {

journey tb_to_journey(schedule const& sched, tb_journey const& tbj);

}  // namespace motis::tripbased
