#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

void write_broken_interchanges_report(paxmon_data const& data,
                                      std::string const& filename);

}  // namespace motis::paxmon
