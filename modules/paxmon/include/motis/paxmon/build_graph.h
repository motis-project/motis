#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct build_graph_stats {
  std::uint64_t groups_not_added_{};
};

void add_passenger_group_to_graph(schedule const& sched,
                                  capacity_maps const& caps, universe& uv,
                                  passenger_group& grp);

void remove_passenger_group_from_graph(universe& uv, passenger_group* pg);

build_graph_stats build_graph_from_journeys(schedule const& sched,
                                            capacity_maps const& caps,
                                            universe& uv);

}  // namespace motis::paxmon
