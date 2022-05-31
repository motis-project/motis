#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/mcraptor/raptor_timetable.h"

namespace motis::mcraptor {

std::pair<std::unique_ptr<raptor_meta_info>, std::unique_ptr<raptor_timetable>>
get_raptor_timetable(schedule const& sched);

}  // namespace motis::mcraptor
