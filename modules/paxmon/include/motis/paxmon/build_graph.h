#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

struct build_graph_stats {
  std::uint64_t initial_over_capacity_{};
  std::uint64_t groups_not_added_{};
};

build_graph_stats build_graph_from_journeys(schedule const& sched,
                                            paxmon_data& data);

}  // namespace motis::paxmon
