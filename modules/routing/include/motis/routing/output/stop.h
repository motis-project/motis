#pragma once

#include "motis/string.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timestamp_reason.h"

namespace motis::routing::output::intermediate {

struct stop {
  unsigned index_;
  unsigned station_id_;
  mcd::string const *a_track_, *d_track_;
  mcd::string const *a_sched_track_, *d_sched_track_;
  time a_time_, d_time_;
  time a_sched_time_, d_sched_time_;
  timestamp_reason a_reason_, d_reason_;
  bool exit_, enter_;
};

}  // namespace motis::routing::output::intermediate
