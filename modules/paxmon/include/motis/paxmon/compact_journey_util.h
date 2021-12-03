#pragma once

#include <optional>
#include <utility>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

compact_journey get_prefix(schedule const& sched, compact_journey const& cj,
                           passenger_localization const& loc);

std::pair<compact_journey, time> get_prefix_and_arrival_time(
    schedule const& sched, compact_journey const& cj,
    unsigned const search_station, time const earliest_arrival);

compact_journey get_suffix(schedule const& sched, compact_journey const& cj,
                           passenger_localization const& loc);

compact_journey merge_journeys(schedule const& sched,
                               compact_journey const& prefix,
                               compact_journey const& suffix);

std::optional<unsigned> get_first_long_distance_station_id(
    universe const& uv, compact_journey const& cj);

std::optional<unsigned> get_last_long_distance_station_id(
    universe const& uv, compact_journey const& cj);

}  // namespace motis::paxmon
