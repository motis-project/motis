#pragma once

#include <cstdint>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "motis/hash_map.h"

#include "motis/paxforecast/alternatives.h"

#include "motis/paxmon/additional_group.h"
#include "motis/paxmon/index_types.h"
#include "motis/paxmon/universe.h"

namespace motis::paxforecast {

struct group_simulation_result {
  motis::paxmon::passenger_localization const* localization_{};
  std::vector<std::pair<alternative const*, float>> alternative_probabilities_;
};

struct simulation_result_stats {
  double found_alt_count_avg_{};
  double picked_alt_count_avg_{};
  double best_alt_prob_avg_{};
  double second_alt_prob_avg_{};
  std::uint64_t group_route_count_{};
  std::uint64_t combined_group_count_{};
};

struct simulation_result {
  mcd::hash_map<motis::paxmon::edge const*,
                std::vector<motis::paxmon::additional_group>>
      additional_groups_;
  mcd::hash_map<motis::paxmon::passenger_group_with_route,
                group_simulation_result>
      group_route_results_;
  simulation_result_stats stats_;
};

}  // namespace motis::paxforecast
