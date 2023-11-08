#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/universe.h"

namespace motis::paxforecast {

struct load_forecast {
  std::vector<motis::paxmon::trip_load_info> trips_;
};

load_forecast calc_load_forecast(schedule const& sched,
                                 motis::paxmon::universe const& uv);

}  // namespace motis::paxforecast
