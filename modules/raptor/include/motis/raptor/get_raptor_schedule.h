#pragma once

#include "motis/core/schedule/footpath.h"
#include "motis/core/schedule/schedule.h"

#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

struct transformable_footpath {
  transformable_footpath(station_id const from, station_id const to,
                         time const duration)
      : from_(from), to_(to), duration_(duration) {}

  station_id from_;
  station_id to_;
  time duration_;
};

struct transformable_stop {
  std::vector<transformable_footpath> footpaths_;
  std::vector<transformable_footpath> incoming_footpaths_;
  std::vector<route_id> stop_routes_;
  std::vector<station_id> equivalent_;
  time transfer_time_{invalid<time>};
  std::string eva_{""};
  unsigned motis_station_index_{invalid<unsigned>};
};

struct raptor_lcon {
  raptor_lcon(station_id const from, station_id const to, time const dep,
              time const arrival, bool const in_allowed, bool const out_allowed,
              light_connection const* lc)
      : from_(from),
        to_(to),
        departure_(dep),
        arrival_(arrival),
        in_allowed_(in_allowed),
        out_allowed_(out_allowed),
        lcon_(lc) {}
  station_id from_;
  station_id to_;
  time departure_;
  time arrival_;
  bool in_allowed_;
  bool out_allowed_;
  light_connection const* lcon_;
};

struct transformable_trip {
  std::vector<raptor_lcon> lcons_;
  std::vector<stop_time> stop_times_;
};

struct transformable_route {
  std::vector<transformable_trip> trips_;
  std::vector<station_id> route_stops_;
  time stand_time_{invalid<time>};
};

struct transformable_timetable {
  std::vector<transformable_stop> stations_;
  std::vector<transformable_route> routes_;
};

std::tuple<std::unique_ptr<raptor_schedule>, std::unique_ptr<raptor_timetable>,
           std::unique_ptr<raptor_timetable>>
get_raptor_schedule(schedule const& sched);
}  // namespace motis::raptor
