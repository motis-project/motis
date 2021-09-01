#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void write_broken_interchanges_report(universe const& uv,
                                      std::string const& filename);

}  // namespace motis::paxmon
