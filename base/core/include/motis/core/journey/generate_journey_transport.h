#pragma once

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

namespace motis {

journey::transport generate_journey_transport(
    unsigned int from, unsigned int to, connection_info const* con_info,
    schedule const& sched, duration duration = 0, int mumo_id = -1,
    unsigned mumo_price = 0, unsigned mumo_accessibility = 0);

}  // namespace motis
