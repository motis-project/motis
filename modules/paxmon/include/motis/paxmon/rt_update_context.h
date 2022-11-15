#pragma once

#include <set>

#include "motis/core/schedule/trip.h"

#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

struct rt_update_context {
  std::set<passenger_group_with_route> group_routes_affected_by_last_update_;
};

}  // namespace motis::paxmon
