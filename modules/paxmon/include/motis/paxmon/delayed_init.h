#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

struct delay_init_options {
  bool reroute_unmatched_{};
  std::string initial_reroute_router_;
};

void delayed_init(paxmon_data& data, universe& uv, schedule const& sched,
                  delay_init_options const& opt);

}  // namespace motis::paxmon
