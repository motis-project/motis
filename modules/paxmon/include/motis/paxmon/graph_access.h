#pragma once

#include <functional>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/module/message.h"

#include "motis/paxmon/capacity_data.h"
#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/statistics.h"

namespace motis::paxmon {

trip_data_index get_or_add_trip(schedule const& sched, paxmon_data& data,
                                trip const* trp);

trip_data_index get_or_add_trip(schedule const& sched, paxmon_data& data,
                                extern_trip const& et);

void update_event_times(schedule const& sched, graph& g,
                        motis::rt::RtDelayUpdate const* du,
                        std::vector<edge_index>& updated_interchange_edges,
                        system_statistics& system_stats);

void update_trip_route(schedule const& sched, paxmon_data& data,
                       motis::rt::RtRerouteUpdate const* ru,
                       std::vector<edge_index>& updated_interchange_edges,
                       system_statistics& system_stats);

inline edge* add_edge(graph& g, edge&& e) {
  return &g.graph_.push_back_edge(std::move(e));
}

inline edge make_trip_edge(graph& g, event_node_index from, event_node_index to,
                           edge_type type, merged_trips_idx merged_trips,
                           std::uint16_t encoded_capacity,
                           service_class clasz) {
  return edge{from,
              to,
              type,
              false,
              0,
              encoded_capacity,
              clasz,
              merged_trips,
              g.pax_connection_info_.insert()};
}

inline edge make_interchange_edge(event_node_index from, event_node_index to,
                                  duration transfer_time, pci_index pci) {
  return edge{from,
              to,
              edge_type::INTERCHANGE,
              false,
              transfer_time,
              UNLIMITED_ENCODED_CAPACITY,
              service_class::OTHER,
              0,
              pci};
}

inline edge make_through_edge(graph& g, event_node_index from,
                              event_node_index to) {
  return edge{from,
              to,
              edge_type::THROUGH,
              false,
              0,
              UNLIMITED_ENCODED_CAPACITY,
              service_class::OTHER,
              0,
              g.pax_connection_info_.insert()};
}

void add_passenger_group_to_edge(graph& g, edge* e, passenger_group* pg);
void remove_passenger_group_from_edge(graph& g, edge* e, passenger_group* pg);

void for_each_trip(
    schedule const& sched, paxmon_data& data, compact_journey const& journey,
    std::function<void(journey_leg const&, trip_data_index)> const& fn);

void for_each_edge(schedule const& sched, paxmon_data& data,
                   compact_journey const& journey,
                   std::function<void(journey_leg const&, edge*)> const& fn);

event_node* find_event_node(graph const& g, trip_data_index tdi,
                            std::uint32_t station_idx, event_type et,
                            time schedule_time);

}  // namespace motis::paxmon
