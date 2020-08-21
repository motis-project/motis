#pragma once

#include <cstddef>
#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/loader/loader_result.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::loader::journeys {

void load_journey(schedule const& sched, paxmon_data& data, journey const& j,
                  data_source const& source, std::uint16_t passengers,
                  group_source_flags source_flags = group_source_flags::NONE);

loader_result load_journeys(schedule const& sched, paxmon_data& data,
                            std::string const& journey_file);

}  // namespace motis::paxmon::loader::journeys
