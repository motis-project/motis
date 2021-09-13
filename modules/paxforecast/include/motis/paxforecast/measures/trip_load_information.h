#pragma once

#include <cstdint>

#include "motis/core/journey/extern_trip.h"

#include "motis/paxforecast/measures/interval.h"
#include "motis/paxforecast/measures/recipients.h"

namespace motis::paxforecast::measures {

enum load_level : std::uint8_t { LOW, NO_SEATS, FULL };

struct trip_load_information {
  recipients recipients_;
  interval interval_;
  extern_trip trip_{};
  load_level level_{};
};

}  // namespace motis::paxforecast::measures
