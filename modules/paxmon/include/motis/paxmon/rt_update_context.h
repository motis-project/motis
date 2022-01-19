#pragma once

#include <set>

#include "motis/core/schedule/trip.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct rt_update_context {
  std::set<passenger_group_index> groups_affected_by_last_update_;
};

}  // namespace motis::paxmon
