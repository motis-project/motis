#pragma once

#include <cstddef>
#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::loader::csv {

std::size_t load_journeys(schedule const& sched, paxmon_data& data,
                          std::string const& journey_file,
                          std::string const& match_log_file,
                          duration match_tolerance);

}  // namespace motis::paxmon::loader::csv
