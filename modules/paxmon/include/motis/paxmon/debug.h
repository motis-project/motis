#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/graph.h"

namespace motis::paxmon {

void print_trip(trip const* trp);

void print_leg(schedule const& sched, journey_leg const& leg);

void print_trip_section(schedule const& sched,
                        motis::access::trip_section const& ts);

void print_trip_edge(schedule const& sched, graph const& g, edge const* e);

void print_trip_sections(graph const& g, schedule const& sched, trip const* trp,
                         trip_data const* td);

}  // namespace motis::paxmon
