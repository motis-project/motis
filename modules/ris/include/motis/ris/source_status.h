#pragma once

#include <cstdint>

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/unixtime.h"

namespace motis::ris {

struct source_status {
  bool enabled_{};
  std::uint32_t update_interval_{};  // seconds

  void add_update(std::uint64_t const messages,
                  unixtime const last_message_time) {
    last_update_time_ = now();
    last_update_messages_ = messages;
    last_message_time_ = last_message_time;
    ++total_updates_;
    total_messages_ += messages;
  }

  // last update/tick
  unixtime last_update_time_{};
  std::uint64_t last_update_messages_{};
  unixtime last_message_time_{};

  // totals
  std::uint64_t total_updates_{};
  std::uint64_t total_messages_{};
};

}  // namespace motis::ris
