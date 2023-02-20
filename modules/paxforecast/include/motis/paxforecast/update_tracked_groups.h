#pragma once

#include <map>
#include <optional>

#include "motis/hash_map.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/broken_transfer_info.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/universe.h"

#include "motis/paxforecast/simulation_result.h"
#include "motis/paxforecast/statistics.h"

namespace motis::paxforecast {

void update_tracked_groups(
    schedule const& sched, motis::paxmon::universe& uv,
    simulation_result const& sim_result,
    std::map<motis::paxmon::passenger_group_with_route,
             motis::paxmon::monitoring_event_type> const& pgwr_event_types,
    std::map<motis::paxmon::passenger_group_with_route,
             std::optional<motis::paxmon::broken_transfer_info>> const&
        broken_transfer_infos,
    mcd::hash_map<motis::paxmon::passenger_group_with_route,
                  motis::paxmon::passenger_localization const*> const&
        pgwr_localizations,
    tick_statistics& tick_stats,
    motis::paxmon::reroute_reason_t const default_reroute_reason);

}  // namespace motis::paxforecast
