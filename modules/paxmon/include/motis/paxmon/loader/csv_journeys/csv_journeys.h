#pragma once

#include <cstddef>
#include <cstdint>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/loader/loader_result.h"
#include "motis/paxmon/settings/journey_input_settings.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon::loader::csv_journeys {

loader_result load_journeys(
    schedule const& sched, universe& uv, std::string const& journey_file,
    motis::paxmon::settings::journey_input_settings const& settings);

}  // namespace motis::paxmon::loader::csv_journeys
