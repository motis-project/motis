#pragma once

#include <vector>

#include "boost/filesystem.hpp"

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon::output {

void write_scenario(boost::filesystem::path const& dir, schedule const& sched,
                    capacity_maps const& caps, universe const& uv,
                    std::vector<motis::module::msg_ptr> const& messages,
                    bool include_trip_info = false);

}  // namespace motis::paxmon::output
