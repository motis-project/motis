#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/revise/section.h"
#include "motis/revise/stop.h"

namespace motis::revise {

std::vector<section> get_sections(journey const& j);
std::vector<stop_ptr> get_all_stops(schedule const& sched, journey const& j);

}  // namespace motis::revise
