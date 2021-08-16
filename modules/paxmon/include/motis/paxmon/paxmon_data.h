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

};

}  // namespace motis::paxmon
