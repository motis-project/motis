#pragma once

#include <map>
#include <string_view>
#include <vector>

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/rt/delay_propagator.h"
#include "motis/rt/message_history.h"
#include "motis/rt/metrics.h"
#include "motis/rt/schedule_event.h"
#include "motis/rt/statistics.h"
#include "motis/rt/update_msg_builder.h"

namespace motis::rt {

struct full_trip_result {
  enum class status {
    OK,
    INVALID_STATION_SEQUENCE,
    INVALID_SCHEDULE_TIME_SEQUENCE
  };

  std::vector<node_id_t> stations_addded_;
  trip* trp_{};
  status status_{status::OK};
  bool is_new_trip_{};
  bool is_reroute_{};
  unsigned delay_updates_{};
  unsigned track_updates_{};
};

full_trip_result handle_full_trip_msg(
    statistics& stats, schedule& sched, update_msg_builder& update_builder,
    message_history& msg_history, delay_propagator& propagator,
    ris::FullTripMessage const* msg, std::string_view msg_buffer,
    std::map<schedule_event, delay_info*>& cancelled_delays,
    rt_metrics& metrics, unixtime msg_timestamp, unixtime processing_time);

}  // namespace motis::rt
