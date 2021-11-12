#pragma once

#include "motis/raptor/criteria/criteria_config.h"
#include "motis/raptor/criteria/traits.h"
#include "motis/raptor/criteria/traits/max_occupancy.h"

namespace motis::raptor {

using Default = criteria_config<traits<>>;
using MaxOccupancy = criteria_config<traits<trait_max_occupancy>>;

}  // namespace motis::raptor
