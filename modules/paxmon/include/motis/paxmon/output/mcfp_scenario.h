#pragma once

#include <vector>

#include "boost/filesystem.hpp"

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon::output {

void write_scenario(boost::filesystem::path const& dir, schedule const& sched,
                    paxmon_data const& data,
                    std::vector<motis::module::msg_ptr> const& messages,
                    bool include_trip_info = false);

}  // namespace motis::paxmon::output
