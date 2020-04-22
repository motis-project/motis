#pragma once

#include <map>
#include <vector>

#include "motis/vector.h"

#include "motis/core/schedule/schedule.h"

namespace motis::railviz {

struct path_resolver {
  path_resolver(schedule const& sched, int zoom_level);

  std::vector<std::vector<double>> get_trip_path(trip const* trp);
  std::pair<bool, std::vector<double>> get_segment_path(edge const* e);

  int get_req_count() const { return req_count_; }

private:
  schedule const& sched_;
  unsigned zoom_level_;
  std::map<mcd::vector<trip::route_edge> const*,
           std::vector<std::vector<double>>>
      trip_cache_;
  std::map<edge const*, std::pair<bool, std::vector<double>>> edge_cache_;
  int req_count_;
};

}  // namespace motis::railviz
