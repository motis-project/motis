#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxforecast/simulation_result.h"
#include "motis/paxmon/get_load.h"
#include "motis/paxmon/graph.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxforecast {

struct load_forecast {
  std::vector<motis::paxmon::trip_load_info> trips_;
};

load_forecast calc_load_forecast(schedule const& sched,
                                 motis::paxmon::paxmon_data const& data,
                                 simulation_result const& sim_result);

}  // namespace motis::paxforecast
