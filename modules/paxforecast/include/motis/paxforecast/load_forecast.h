#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/paxforecast/simulation_result.h"
#include "motis/paxmon/get_load.h"
#include "motis/paxmon/graph.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxforecast {

struct edge_forecast {
  motis::paxmon::edge const* edge_{};
  motis::paxmon::pax_cdf forecast_cdf_;
  bool updated_{};
  bool possibly_over_capacity_{};
  std::uint16_t expected_passengers_{};
};

struct trip_forecast {
  trip const* trp_{};
  std::vector<edge_forecast> edges_;
};

struct load_forecast {
  std::vector<trip_forecast> trips_;
};

load_forecast calc_load_forecast(schedule const& sched,
                                 motis::paxmon::paxmon_data const& data,
                                 simulation_result const& sim_result);

}  // namespace motis::paxforecast
