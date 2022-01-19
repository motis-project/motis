#pragma once

#include <ctime>
#include <string>

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/journey/journey.h"

namespace motis::revise {

ev_key get_ev_key(schedule const& sched, journey const& j, unsigned stop_idx,
                  event_type ev_type);

ev_key get_ev_key_from_trip(schedule const& sched, trip const* trp,
                            std::string const& station_id,
                            event_type const& ev_type, unixtime schedule_time);

}  // namespace motis::revise
