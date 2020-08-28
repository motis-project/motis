#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/rt/delay_propagator.h"
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
  trip const* trp_{};
  status status_{status::OK};
  bool is_new_trip_{};
  bool is_reroute_{};
  unsigned delay_updates_{};
};

full_trip_result handle_full_trip_msg(statistics& stats, schedule& sched,
                                      update_msg_builder& update_builder,
                                      delay_propagator& propagator,
                                      ris::FullTripMessage const* msg);

}  // namespace motis::rt
