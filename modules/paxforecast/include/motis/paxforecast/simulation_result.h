#pragma once

#include <cstdint>
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

struct simulation_result_stats {
  double found_alt_count_avg_{};
  double picked_alt_count_avg_{};
  double best_alt_prob_avg_{};
  double second_alt_prob_avg_{};
  std::uint64_t group_count_{};
  std::uint64_t combined_group_count_{};
};

struct simulation_result {
  mcd::hash_map<
      motis::paxmon::edge const*,
      std::vector<std::pair<motis::paxmon::passenger_group const*, float>>>
      additional_groups_;
  mcd::hash_map<motis::paxmon::passenger_group const*, group_simulation_result>
      group_results_;
  simulation_result_stats stats_;
};

}  // namespace motis::paxforecast
