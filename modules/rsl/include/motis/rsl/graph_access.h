#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/module/message.h"

#include "motis/rsl/rsl_data.h"
#include "motis/rsl/statistics.h"

namespace motis::rsl {

trip_data* get_or_add_trip(schedule const& sched, rsl_data& data,
                           extern_trip const& et);

void update_event_times(schedule const& sched, graph& g,
                        motis::rt::RtDelayUpdate const* du,
                        std::vector<edge*>& updated_interchange_edges,
                        system_statistics& system_stats);

void update_trip_route(schedule const& sched, rsl_data& data,
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

}  // namespace motis::rsl
