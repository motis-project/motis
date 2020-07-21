#pragma once

#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "motis/hash_map.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxmon/graph.h"

namespace motis::paxforecast {

struct group_simulation_result {
  motis::paxmon::passenger_localization const* localization_{};
  std::vector<std::pair<alternative const*, float>> alternatives_;
};

struct simulation_result {
  mcd::hash_map<
      motis::paxmon::edge const*,
      std::vector<std::pair<motis::paxmon::passenger_group const*, float>>>
      additional_groups_;
  mcd::hash_map<motis::paxmon::passenger_group const*, group_simulation_result>
      group_results_;
};

}  // namespace motis::paxforecast
