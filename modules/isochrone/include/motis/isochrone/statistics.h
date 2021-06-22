#pragma once

#include "motis/core/statistics/statistics.h"

namespace motis::isochrone {

struct statistics {
  uint64_t start_edges_{};
  uint64_t time_dependent_dijkstra_{};
  uint64_t num_of_results_{};
};

inline stats_category to_stats_category(char const* name, statistics const& s) {
  return {
      name,
      {{"start_edges", s.start_edges_},
              {"time_dependent_dijkstra", s.time_dependent_dijkstra_},
              {"num_of_results", s.num_of_results_}}};
}

}  // namespace motis::intermodal
