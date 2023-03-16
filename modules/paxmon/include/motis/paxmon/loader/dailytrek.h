#pragma once

#include <string>
#include <vector>

#include "motis/core/schedule/schedule.h"

namespace motis::paxmon::loader {

std::vector<std::string> get_dailytrek_files(schedule const& sched,
                                             std::string const& dir);

}  // namespace motis::paxmon::loader
