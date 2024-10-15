#pragma once

#include <memory>

#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

using gpu_route = raptor_route;
using gpu_stop = raptor_stop;

struct gpu_footpath {
  gpu_footpath()
      : from_{invalid<decltype(from_)>},
        to_{-1},
        duration_{invalid<decltype(duration_)>} {}

  gpu_footpath(stop_id const from, stop_id const to, motis::time const duration)
      : from_{from}, to_{to}, duration_{static_cast<time8>(duration)} {
    utl::verify(duration < std::numeric_limits<time8>::max(),
                "Footpath duration too long to fit inside time8");
  }

  stop_id from_;
  stop_id to_ : 24;
  time8 duration_;
};

struct host_gpu_timetable {
  host_gpu_timetable() = default;

  // subtract the sentinel
  auto stop_count() const { return stops_.size() - 1; }
  auto route_count() const { return routes_.size() - 1; }

  std::vector<gpu_stop> stops_;
  std::vector<gpu_route> routes_;
  std::vector<gpu_footpath> footpaths_;

  std::vector<stop_id> route_stops_;
  std::vector<route_id> stop_routes_;

  std::vector<motis::time> stop_departures_;
  std::vector<motis::time> stop_arrivals_;
  std::vector<occ_t> stop_inb_occupancy_;

  std::vector<time> transfer_times_;
};

struct device_gpu_timetable {
  gpu_stop* stops_;
  gpu_route* routes_;

  gpu_footpath* footpaths_;

  time* transfer_times_;

  motis::time* stop_arrivals_;
  motis::time* stop_departures_;
  occ_t* stop_inb_occupancy_;

  stop_id* route_stops_;

  route_id* stop_routes_;

  stop_id stop_count_;
  route_id route_count_;
  footpath_id footpath_count_;
};

std::unique_ptr<host_gpu_timetable> get_host_gpu_timetable(
    raptor_timetable const& tt);

std::unique_ptr<device_gpu_timetable> get_device_gpu_timetable(
    host_gpu_timetable const& h_gtt);

void destroy_device_gpu_timetable(device_gpu_timetable& d_gtt);

}  // namespace motis::raptor