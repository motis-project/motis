#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

std::pair<std::unique_ptr<raptor_schedule>, std::unique_ptr<raptor_timetable>>
get_raptor_schedule(schedule const& sched);

}  // namespace motis::raptor
