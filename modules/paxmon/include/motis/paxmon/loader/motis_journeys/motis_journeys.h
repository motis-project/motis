#pragma once

#include <cstddef>
#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/loader/loader_result.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon::loader::motis_journeys {

void load_journey(schedule const& sched, universe& uv, journey const& j,
                  data_source const& source, std::uint16_t passengers,
                  route_source_flags source_flags = route_source_flags::NONE);

loader_result load_journeys(schedule const& sched, universe& uv,
                            std::string const& journey_file);

}  // namespace motis::paxmon::loader::motis_journeys
