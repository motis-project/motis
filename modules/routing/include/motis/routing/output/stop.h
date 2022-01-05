#pragma once

#include "motis/string.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timestamp_reason.h"

namespace motis::routing::output::intermediate {

struct stop {
  unsigned index_{0U};
  unsigned station_id_{0U};
  mcd::string const *a_track_{nullptr}, *d_track_{nullptr};
  mcd::string const *a_sched_track_{nullptr}, *d_sched_track_{nullptr};
  time a_time_, d_time_;
  time a_sched_time_, d_sched_time_;
  timestamp_reason a_reason_{timestamp_reason::SCHEDULE},
      d_reason_{timestamp_reason::SCHEDULE};
  bool exit_{false}, enter_{false};
};

}  // namespace motis::routing::output::intermediate
