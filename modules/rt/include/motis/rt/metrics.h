#pragma once

#include <cstdint>

#include "motis/core/common/unixtime.h"
#include "motis/core/metrics/metrics_storage.h"

#include "motis/module/message.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::rt {

struct metrics_entry {
  std::uint32_t messages_{};
  std::uint32_t delay_messages_{};
  std::uint32_t cancel_messages_{};
  std::uint32_t additional_messages_{};
  std::uint32_t reroute_messages_{};
  std::uint32_t track_messages_{};
  std::uint32_t full_trip_messages_{};
  std::uint32_t trip_formation_messages_{};

  // full trip messages:
  std::uint32_t ft_schedule_messages_{};
  std::uint32_t ft_update_messages_{};
  std::uint32_t ft_new_trips_{};
  std::uint32_t ft_cancellations_{};
  std::uint32_t ft_reroutes_{};
  std::uint32_t ft_rule_service_reroutes_{};
  std::uint32_t ft_trip_delay_updates_{};
  std::uint32_t ft_event_delay_updates_{};
  std::uint32_t ft_trip_track_updates_{};
  std::uint32_t ft_trip_id_not_found_{};
  std::uint32_t ft_trip_id_ambiguous_{};

  // trip formation messages:
  std::uint32_t formation_schedule_messages_{};
  std::uint32_t formation_preview_messages_{};
  std::uint32_t formation_is_messages_{};
  std::uint32_t formation_invalid_primary_trip_id_{};
  std::uint32_t formation_primary_trip_id_not_found_{};
  std::uint32_t formation_primary_trip_id_ambiguous_{};
};

using rt_metrics_storage = metrics_storage<metrics_entry>;

struct rt_metrics {
  template <typename Fn>
  inline void update(unixtime const msg_timestamp,
                     unixtime const processing_time, Fn&& fn) {
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

motis::module::msg_ptr get_metrics_api(rt_metrics const& metrics);

}  // namespace motis::rt
