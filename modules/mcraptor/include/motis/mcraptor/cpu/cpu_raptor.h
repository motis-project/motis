#pragma once

#include "motis/mcraptor/cpu/mark_store.h"

#include "motis/mcraptor/raptor_query.h"
#include "motis/mcraptor/raptor_result.h"
#include "motis/mcraptor/raptor_statistics.h"
#include "motis/mcraptor/raptor_timetable.h"

namespace motis::mcraptor {

trip_count get_earliest_trip(raptor_timetable const& tt,
                             raptor_route const& route,
                             time const* prev_arrivals,
                             stop_times_index r_stop_offset);

void init_arrivals(raptor_result& result, raptor_query const& q,
                   cpu_mark_store& station_marks);

void update_route(raptor_timetable const& tt, route_id r_id,
                  time const* prev_arrivals, time* current_round,
                  earliest_arrivals& ea, cpu_mark_store& station_marks);

void update_footpaths(raptor_timetable const& tt, time* current_round,
                      earliest_arrivals const& ea,
                      cpu_mark_store& station_marks);

void invoke_cpu_raptor(raptor_query const& query, raptor_statistics&);

}  // namespace motis::mcraptor