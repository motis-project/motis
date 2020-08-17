#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxforecast/simulation_result.h"
#include "motis/paxmon/graph.h"

namespace motis::paxforecast {

struct edge_over_capacity_info {
  std::uint16_t current_pax_{};
  std::uint16_t additional_pax_{};
};

struct over_capacity_info {
  float probability_{};
  mcd::hash_map<motis::paxmon::edge const*, edge_over_capacity_info>
      over_capacity_edges_;
  mcd::hash_map<trip const*, std::vector<motis::paxmon::edge const*>>
      over_capacity_trips_;
};

over_capacity_info calc_over_capacity(schedule const& sched,
                                      simulation_result const& sim_result,
                                      float probability);

}  // namespace motis::paxforecast
