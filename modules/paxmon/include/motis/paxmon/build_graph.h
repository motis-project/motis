#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

struct build_graph_stats {
  std::uint64_t groups_not_added_{};
};

void add_passenger_group_to_graph(schedule const& sched, paxmon_data& data,
                                  passenger_group& grp);

void remove_passenger_group_from_graph(passenger_group* pg);

build_graph_stats build_graph_from_journeys(schedule const& sched,
                                            paxmon_data& data);

}  // namespace motis::paxmon
