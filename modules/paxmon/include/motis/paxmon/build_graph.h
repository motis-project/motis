#pragma once

#include <cstdint>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct add_group_route_to_graph_result {
  inline bool has_valid_times() const {
    return scheduled_arrival_time_ != INVALID_TIME &&
           current_arrival_time_ != INVALID_TIME;
  }

  motis::time scheduled_arrival_time_{INVALID_TIME};
  motis::time current_arrival_time_{INVALID_TIME};
};

add_group_route_to_graph_result add_group_route_to_graph(
    schedule const& sched, capacity_maps const& caps, universe& uv,
    passenger_group const& grp, group_route const& gr, bool log,
    pci_log_reason_t reason);

void remove_group_route_from_graph(universe& uv, schedule const& sched,
                                   passenger_group const& grp,
                                   group_route const& gr, bool log,
                                   pci_log_reason_t reason);

}  // namespace motis::paxmon
