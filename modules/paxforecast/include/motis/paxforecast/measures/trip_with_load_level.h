#pragma once

#include "motis/core/journey/extern_trip.h"

#include "motis/paxforecast/measures/load_level.h"

namespace motis::paxforecast::measures {

struct trip_with_load_level {
  extern_trip trip_{};
  load_level level_{load_level::UNKNOWN};
};

}  // namespace motis::paxforecast::measures
