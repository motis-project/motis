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

inline edge make_trip_edge(event_node* from, event_node* to, edge_type type,
                           trip const* trp, std::uint16_t capacity) {
  return edge{from, to, type, false, 0, capacity, 0, trp, {}};
}

inline edge make_interchange_edge(event_node* from, event_node* to,
                                  duration transfer_time,
                                  std::uint16_t passengers,
                                  pax_connection_info&& ci) {
  return edge{
      from,       to,      edge_type::INTERCHANGE, false, transfer_time, 0,
      passengers, nullptr, std::move(ci)};
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
