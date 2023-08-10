#pragma once

#include "motis/core/common/unixtime.h"
#include "motis/core/metrics/metrics_storage.h"

namespace motis::paxmon {

template <typename T>
struct metrics {
  inline void add(unixtime const system_time, unixtime const processing_time,
                  T const& new_metrics) {
    if (auto* m = by_system_time_.at(system_time); m != nullptr) {
      *m += new_metrics;
    }
    if (auto* m = by_processing_time_.at(processing_time); m != nullptr) {
      *m += new_metrics;
    }
  }

  metrics_storage<T> by_system_time_;
  metrics_storage<T> by_processing_time_;
};

}  // namespace motis::paxmon
