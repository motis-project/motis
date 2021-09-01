#pragma once

#include "motis/paxmon/capacity.h"

namespace motis::paxmon {

struct capacity_maps {
  trip_capacity_map_t trip_capacity_map_;
  category_capacity_map_t category_capacity_map_;
};

}  // namespace motis::paxmon
