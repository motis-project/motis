#pragma once

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timestamp_reason.h"

namespace motis::routing::output::intermediate {

struct stop {
  stop() = default;
  stop(unsigned int index, unsigned int station_id, unsigned int a_track,
       unsigned int d_track, unsigned int a_sched_track,
       unsigned int d_sched_track, time a_time, time d_time, time a_sched_time,
       time d_sched_time, timestamp_reason a_reason, timestamp_reason d_reason,
       bool exit, bool enter)
      : index_(index),
        station_id_(station_id),
        a_track_(a_track),
        d_track_(d_track),
        a_sched_track_(a_sched_track),
        d_sched_track_(d_sched_track),
        a_time_(a_time),
        d_time_(d_time),
        a_sched_time_(a_sched_time),
        d_sched_time_(d_sched_time),
        a_reason_(a_reason),
        d_reason_(d_reason),
        exit_(exit),
        enter_(enter) {}

  unsigned int index_;
  unsigned int station_id_;
  unsigned int a_track_, d_track_;
  unsigned int a_sched_track_, d_sched_track_;
  time a_time_, d_time_;
  time a_sched_time_, d_sched_time_;
  timestamp_reason a_reason_, d_reason_;
  bool exit_, enter_;
};

}  // namespace motis::routing::output::intermediate
