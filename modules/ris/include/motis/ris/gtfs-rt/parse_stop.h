#pragma once

#include <ctime>

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event_type.h"

#include "gtfsrt.pb.h"

namespace motis {

struct trip;
struct schedule;

namespace ris::gtfsrt {

struct known_stop_skips;

struct stop_context {
  void update(schedule const&, trip const&,
              transit_realtime::TripUpdate_StopTimeUpdate const&,
              known_stop_skips*);

  std::string station_id_;
  unsigned seq_no_{std::numeric_limits<unsigned>::max()};
  int idx_{std::numeric_limits<int>::max()};
  bool is_skip_known_{false};

  unixtime stop_arrival_{0};
  unixtime stop_departure_{0};
};

int get_stop_edge_idx(int, event_type);

std::string parse_stop_id(std::string const&);

}  // namespace ris::gtfsrt
}  // namespace motis
