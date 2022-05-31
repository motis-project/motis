#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void print_trip(schedule const& sched, trip const* trp);

void print_trip(schedule const& sched, trip_idx_t idx);

void print_leg(schedule const& sched, journey_leg const& leg);

void print_trip_section(schedule const& sched,
                        motis::access::trip_section const& ts);

void print_trip_edge(schedule const& sched, universe const& uv, edge const* e);

void print_trip_sections(universe const& uv, schedule const& sched,
                         trip const* trp, trip_data_index tdi);

void print_trip_sections(universe const& uv, schedule const& sched,
                         trip_idx_t idx, trip_data_index tdi);

}  // namespace motis::paxmon
