#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/constant_graph.h"
#include "motis/core/schedule/schedule.h"

#include "motis/tripbased/limits.h"
#include "motis/tripbased/query.h"

namespace motis::tripbased {

struct lower_bounds {
  lower_bounds(constant_graph const& travel_time_graph,
               std::vector<int> const& goals,
               mcd::hash_map<unsigned, std::vector<simple_edge>> const&
                   additional_travel_time_edges)
      : travel_time_(travel_time_graph, goals, additional_travel_time_edges) {}
  constant_graph_dijkstra<MAX_TRAVEL_TIME, map_station_graph_node> travel_time_;
};

lower_bounds calc_lower_bounds(schedule const& sched,
                               trip_based_query const& query);

}  // namespace motis::tripbased
