#pragma once

#include <set>

#include "motis/hash_map.h"

#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

struct rt_update_context {
  void reset() {
    group_routes_affected_by_last_update_.clear();
    previous_broken_status_.clear();
  }

  std::set<passenger_group_with_route> group_routes_affected_by_last_update_;
  mcd::hash_map<passenger_group_with_route, bool> previous_broken_status_;
};

}  // namespace motis::paxmon
