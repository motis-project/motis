#pragma once

#include <map>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/rt/statistics.h"
#include "motis/rt/update_msg_builder.h"

namespace motis::rt {

void handle_trip_formation_msg(statistics& stats, schedule& sched,
                               update_msg_builder& update_builder,
                               ris::TripFormationMessage const* msg);

}  // namespace motis::rt
