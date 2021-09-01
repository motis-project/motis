#pragma once

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/multiverse.h"

namespace motis::paxmon {

struct paxmon_data {
  capacity_maps capacity_maps_;
  multiverse multiverse_;
};

}  // namespace motis::paxmon
