#pragma once

#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/rsl/rsl_data.h"

namespace motis::rsl::loader::journeys {

void load_journeys(schedule const& sched, rsl_data& data,
                   std::string const& journey_file);

}  // namespace motis::rsl::loader::journeys
