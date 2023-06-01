#pragma once

#include <cstdint>

#include "motis/core/common/unixtime.h"
#include "motis/core/metrics/metrics_storage.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::rt {

struct metrics_entry {
  std::uint32_t messages_;
  std::uint32_t delay_messages_;
  std::uint32_t cancel_messages_;
  std::uint32_t additional_messages_;
  std::uint32_t reroute_messages_;
  std::uint32_t track_messages_;
  std::uint32_t full_trip_messages_;
  std::uint32_t trip_formation_messages_;
};

using rt_metrics_storage = metrics_storage<metrics_entry>;

struct rt_metrics {
  template <typename Fn>
  void update(unixtime const msg_timestamp, unixtime const processing_time,
              Fn&& fn) {
    if (auto* m = by_msg_timestamp_.at(msg_timestamp); m != nullptr) {
      fn(m);
    }
    if (auto* m = by_processing_time_.at(processing_time); m != nullptr) {
      fn(m);
    }
  }

  rt_metrics_storage by_msg_timestamp_;
  rt_metrics_storage by_processing_time_;
};

void count_message(rt_metrics& metrics, motis::ris::RISMessage const* msg,
                   unixtime processing_time);

}  // namespace motis::rt
