#pragma once

#include "motis/core/schedule/schedule.h"

namespace motis::access {

class trip_stop {
public:
  trip_stop(trip const* t, int index);

  int index() const;

  bool has_arrival() const;
  bool has_departure() const;

  light_connection const& arr_lcon() const;
  light_connection const& dep_lcon() const;

  ev_key arr() const;
  ev_key dep() const;

  connection_info const& arr_info(schedule const& sched) const;
  connection_info const& dep_info(schedule const& sched) const;
  station const& get_station(schedule const& sched) const;

  uint32_t get_station_id() const;
  node const* get_route_node() const;

  bool is_first() const;
  bool is_last() const;

private:
  trip const* trip_;
  int index_;
  node const* node_;
};

}  // namespace motis::access
