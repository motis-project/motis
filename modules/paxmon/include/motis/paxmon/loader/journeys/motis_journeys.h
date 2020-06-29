#pragma once

#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::loader::journeys {

void load_journeys(schedule const& sched, paxmon_data& data,
                   std::string const& journey_file);

}  // namespace motis::paxmon::loader::journeys
