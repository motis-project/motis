#pragma once

#include <memory>
#include <set>
#include <vector>

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/graph.h"

namespace motis::paxmon {

struct paxmon_data {
  passenger_group const* get_passenger_group(passenger_group_index id) const;

  graph graph_;

  std::set<passenger_group*> groups_affected_by_last_update_;
  std::set<trip const*> trips_affected_by_last_update_;

  trip_capacity_map_t trip_capacity_map_;
  category_capacity_map_t category_capacity_map_;
};

}  // namespace motis::paxmon
