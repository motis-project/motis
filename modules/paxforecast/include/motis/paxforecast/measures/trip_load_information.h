#pragma once

#include "motis/core/schedule/time.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/paxforecast/measures/load_level.h"
#include "motis/paxforecast/measures/recipients.h"

namespace motis::paxforecast::measures {

struct trip_load_information {
  recipients recipients_;
  time time_{};
  extern_trip trip_{};
  load_level level_{load_level::UNKNOWN};
};

}  // namespace motis::paxforecast::measures
