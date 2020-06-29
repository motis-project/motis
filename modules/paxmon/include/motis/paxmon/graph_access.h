#pragma once

#include <functional>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/statistics.h"

namespace motis::paxmon {

trip_data* get_or_add_trip(schedule const& sched, paxmon_data& data,
                           extern_trip const& et);

void update_event_times(schedule const& sched, graph& g,
                        motis::rt::RtDelayUpdate const* du,
                        std::vector<edge*>& updated_interchange_edges,
                        system_statistics& system_stats);

void update_trip_route(schedule const& sched, paxmon_data& data,
                       motis::rt::RtRerouteUpdate const* ru,
                       std::vector<edge*>& updated_interchange_edges,
                       system_statistics& system_stats);

inline edge* add_edge(edge const& e) {
  auto edge_ptr =
      e.from_->out_edges_.emplace_back(std::make_unique<edge>(e)).get();
  e.to_->in_edges_.emplace_back(edge_ptr);
  return edge_ptr;
}

void add_passenger_group_to_edge(edge* e, passenger_group* pg);
void remove_passenger_group_from_edge(edge* e, passenger_group* pg);

void for_each_trip(
    schedule const& sched, paxmon_data& data, compact_journey const& journey,
    std::function<void(journey_leg const&, trip_data const*)> const& fn);

void for_each_edge(schedule const& sched, paxmon_data& data,
                   compact_journey const& journey,
                   std::function<void(journey_leg const&, edge*)> const& fn);

}  // namespace motis::paxmon
