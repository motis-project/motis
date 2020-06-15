#pragma once

#include "motis/core/schedule/schedule.h"

#include "motis/rsl/rsl_data.h"

namespace motis::rsl {

extern std::uint64_t initial_over_capacity;

void build_graph_from_journeys(schedule const& sched, rsl_data& data);

}  // namespace motis::rsl
