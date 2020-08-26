#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/constant_graph.h"
#include "motis/core/schedule/schedule.h"

#include "motis/routing/label/criteria/transfers.h"
#include "motis/routing/label/criteria/travel_time.h"

namespace motis::routing {

struct lower_bounds {
  lower_bounds(schedule const& sched,  //
               constant_graph const& travel_time_graph,
               constant_graph const& transfers_graph,  //
               std::vector<int> const& goals,
               mcd::hash_map<unsigned, std::vector<simple_edge>> const&
                   additional_travel_time_edges,
               mcd::hash_map<unsigned, std::vector<simple_edge>> const&
                   additional_transfers_edges)
      : travel_time_(travel_time_graph, goals, additional_travel_time_edges),
        transfers_(transfers_graph, goals, additional_transfers_edges,
                   map_interchange_graph_node(sched.non_station_node_offset_)) {
  }

  constant_graph_dijkstra<MAX_TRAVEL_TIME, map_station_graph_node> travel_time_;
  constant_graph_dijkstra<MAX_TRANSFERS, map_interchange_graph_node> transfers_;
};

}  // namespace motis::routing
