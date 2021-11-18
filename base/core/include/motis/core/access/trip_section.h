#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis::access {

struct trip_section {
  trip_section(concrete_trip t, int index);

  int index() const;

  light_connection const& lcon() const;

  connection const& fcon() const;
  connection_info const& info(schedule const& sched) const;

  class edge const* edge() const;

  station const& from_station(schedule const& sched) const;
  station const& to_station(schedule const& sched) const;

  uint32_t from_station_id() const;
  uint32_t to_station_id() const;

  ev_key ev_key_from() const;
  ev_key ev_key_to() const;

  class edge const* get_route_edge() const;

  node* from_node() const;
  node* to_node() const;

  time arr_time() const;
  time dep_time() const;

  concrete_trip ctrp_;
  int index_;
  class edge const* edge_;
};

}  // namespace motis::access
