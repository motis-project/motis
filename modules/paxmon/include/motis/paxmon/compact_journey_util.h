#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

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

std::optional<std::uint16_t> get_arrival_track(schedule const& sched,
                                               trip const* trp,
                                               std::uint32_t exit_station_id,
                                               motis::time exit_time);

std::optional<std::uint16_t> get_arrival_track(schedule const& sched,
                                               journey_leg const& leg);

std::optional<std::uint16_t> get_departure_track(schedule const& sched,
                                                 trip const* trp,
                                                 std::uint32_t enter_station_id,
                                                 motis::time enter_time);

std::optional<std::uint16_t> get_departure_track(schedule const& sched,
                                                 journey_leg const& leg);

}  // namespace motis::paxmon
