#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/module/message.h"

#include "motis/rsl/rsl_data.h"

namespace motis::rsl {

trip_data* get_or_add_trip(schedule const& sched, rsl_data& data,
                           extern_trip const& et);

void update_event_times(schedule const& sched, graph& g,
                        motis::rt::RtDelayUpdate const* du,
                        std::vector<edge*>& updated_interchange_edges);

void update_trip_route(schedule const& sched, rsl_data& data,
                       motis::rt::RtRerouteUpdate const* ru,
                       std::vector<edge*>& updated_interchange_edges);

inline edge* add_edge(edge const& e) {
  auto edge_ptr =
      e.from_->out_edges_.emplace_back(std::make_unique<edge>(e)).get();
  e.to_->in_edges_.emplace_back(edge_ptr);
  return edge_ptr;
}

void add_passenger_group_to_edge(edge* e, passenger_group* pg);
void remove_passenger_group_from_edge(edge* e, passenger_group* pg);

extern std::uint64_t update_event_times_count;
extern std::uint64_t update_event_times_trip_found;
extern std::uint64_t update_event_times_trip_edges_found;
extern std::uint64_t update_event_times_dep_updated;
extern std::uint64_t update_event_times_arr_updated;

extern std::uint64_t total_updated_interchange_edges;

extern std::uint64_t update_trip_route_count;
extern std::uint64_t update_trip_route_trip_found;
extern std::uint64_t update_trip_route_trip_edges_found;

}  // namespace motis::rsl
