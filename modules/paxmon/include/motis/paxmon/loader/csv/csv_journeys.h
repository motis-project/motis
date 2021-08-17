#pragma once

#include <cstddef>
#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/loader/loader_result.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon::loader::csv {

loader_result load_journeys(schedule const& sched, universe& uv,
                            std::string const& journey_file,
                            std::string const& match_log_file,
                            duration match_tolerance);

}  // namespace motis::paxmon::loader::csv
