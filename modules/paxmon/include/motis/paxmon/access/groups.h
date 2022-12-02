#pragma once

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/index_types.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/temp_passenger_group.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

passenger_group* add_passenger_group(
    universe& uv, schedule const& sched, capacity_maps const& caps,
    temp_passenger_group const& tpg, bool log,
    pci_log_reason_t reason = pci_log_reason_t::UNKNOWN);

struct add_group_route_result {
  passenger_group_with_route pgwr_{};
  bool new_route_{};
  float previous_probability_{};
  float new_probability_{};
};

add_group_route_result add_group_route(universe& uv, schedule const& sched,
                                       capacity_maps const& caps,
                                       passenger_group_index pgi,
                                       temp_group_route const& tgr,
                                       bool override_probabilities, bool log,
                                       pci_log_reason_t reason);

void remove_passenger_group(
    universe& uv, schedule const& sched, passenger_group_index pgi, bool log,
    pci_log_reason_t reason = pci_log_reason_t::UNKNOWN);

void remove_group_route(universe& uv, schedule const& sched,
                        passenger_group_with_route pgwr, bool log,
                        pci_log_reason_t reason);

}  // namespace motis::paxmon
