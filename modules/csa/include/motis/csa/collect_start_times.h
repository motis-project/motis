#pragma once

#include <set>

#include "motis/core/schedule/interval.h"
#include "motis/core/schedule/schedule.h"

#include "motis/csa/csa_query.h"
#include "motis/csa/csa_timetable.h"

namespace motis::csa {

std::set<motis::time> collect_start_times(csa_timetable const&,
                                          csa_query const&, interval,
                                          bool ontrip_at_interval_end);

}  // namespace motis::csa
