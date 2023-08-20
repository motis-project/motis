#pragma once

#include <functional>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/module/message.h"

#include "motis/paxmon/capacity_data.h"
#include "motis/paxmon/index_types.h"
#include "motis/paxmon/statistics.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

trip_data_index get_or_add_trip(schedule const& sched, universe& uv,
                                trip_idx_t trip_idx);

trip_data_index get_or_add_trip(schedule const& sched, universe& uv,
                                trip const* trp);

trip_data_index get_or_add_trip(schedule const& sched, universe& uv,
                                extern_trip const& et);

trip_data_index get_trip(universe const& uv, trip_idx_t trip_idx);

int update_event_times(schedule const& sched, universe& uv,
                       motis::rt::RtDelayUpdate const* du,
                       std::vector<edge_index>& updated_interchange_edges);

void update_trip_route(schedule const& sched, universe& uv,
                       motis::rt::RtRerouteUpdate const* ru,
                       std::vector<edge_index>& updated_interchange_edges);

inline edge* add_edge(universe& uv, edge&& e) {
  return &uv.graph_.push_back_edge(e);
}

inline edge make_trip_edge(universe& uv, event_node_index from,
                           event_node_index to, edge_type type,
                           merged_trips_idx merged_trips,
                           std::uint16_t capacity, capacity_source cap_source,
                           service_class clasz) {
  return edge{from,
              to,
              type,
              false,
              0,
              capacity,
              cap_source,
              clasz,
              merged_trips,
              uv.pax_connection_info_.insert()};
}

inline edge make_interchange_edge(event_node_index from, event_node_index to,
                                  duration transfer_time, pci_index pci) {
  return edge{from,
              to,
              edge_type::INTERCHANGE,
              false,
              transfer_time,
              UNLIMITED_CAPACITY,
              capacity_source::UNLIMITED,
              service_class::OTHER,
              0,
              pci};
}

inline edge make_through_edge(universe& uv, event_node_index from,
                              event_node_index to) {
  return edge{from,
              to,
              edge_type::THROUGH,
              false,
              0,
              UNLIMITED_CAPACITY,
              capacity_source::UNLIMITED,
              service_class::OTHER,
              0,
              uv.pax_connection_info_.insert()};
}

bool add_group_route_to_edge(universe& uv, schedule const& sched, edge* e,
                             passenger_group_with_route const& entry, bool log,
                             pci_log_reason_t reason);

bool remove_group_route_from_edge(universe& uv, schedule const& sched, edge* e,
                                  passenger_group_with_route const& entry,
                                  bool log, pci_log_reason_t reason);

bool add_broken_group_route_to_edge(universe& uv, schedule const& sched,
                                    edge* e,
                                    passenger_group_with_route const& entry,
                                    bool log, pci_log_reason_t reason);

bool remove_broken_group_route_from_edge(
    universe& uv, schedule const& sched, edge* e,
    passenger_group_with_route const& entry, bool log, pci_log_reason_t reason);

void for_each_trip(
    schedule const& sched, universe& uv, compact_journey const& journey,
    std::function<void(journey_leg const&, trip_data_index)> const& fn);

void for_each_edge(schedule const& sched, universe& uv,
                   compact_journey const& journey,
                   std::function<void(journey_leg const&, edge*)> const& fn);

event_node* find_event_node(universe const& uv, trip_data_index tdi,
                            std::uint32_t station_idx, event_type et,
                            time schedule_time);

}  // namespace motis::paxmon
