#pragma once

#include <cstddef>
#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity.h"

namespace motis::paxmon::loader::capacities {

std::size_t load_capacities(schedule const& sched,
                            std::string const& capacity_file,
                            capacity_maps& caps,
                            std::string const& match_log_file = "");

}  // namespace motis::paxmon::loader::capacities
