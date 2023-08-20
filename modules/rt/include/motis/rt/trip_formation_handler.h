#pragma once

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/rt/metrics.h"
#include "motis/rt/statistics.h"
#include "motis/rt/update_msg_builder.h"

namespace motis::rt {

void handle_trip_formation_msg(statistics& stats, schedule& sched,
                               update_msg_builder& update_builder,
                               ris::TripFormationMessage const* msg,
                               rt_metrics& metrics, unixtime msg_timestamp,
                               unixtime processing_time);

}  // namespace motis::rt
