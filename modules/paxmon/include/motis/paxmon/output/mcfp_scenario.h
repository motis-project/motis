#pragma once

#include <filesystem>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon::output {

void write_scenario(std::filesystem::path const& dir, schedule const& sched,
                    universe const& uv,
                    std::vector<motis::module::msg_ptr> const& messages,
                    bool include_trip_info = false);

}  // namespace motis::paxmon::output
