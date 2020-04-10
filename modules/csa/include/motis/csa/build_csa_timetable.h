#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/csa/csa_timetable.h"

namespace motis::csa {

std::unique_ptr<csa_timetable> build_csa_timetable(
    schedule const&, bool bridge_zero_duration_connections,
    bool add_footpath_connections);

}  // namespace motis::csa
