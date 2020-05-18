#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis::access {

struct trip_section {
  trip_section(trip const* t, int index);

  int index() const;

  light_connection const& lcon() const;

  connection const& fcon() const;
  connection_info const& info(schedule const& sched) const;

  station const& from_station(schedule const& sched) const;
  station const& to_station(schedule const& sched) const;

  uint32_t from_station_id() const;
  uint32_t to_station_id() const;

  ev_key ev_key_from() const;
  ev_key ev_key_to() const;

  trip const* trip_;
  int index_;
  edge const* edge_;
};

}  // namespace motis::access
