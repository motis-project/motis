#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/time.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/paxforecast/measures/recipients.h"

namespace motis::paxforecast::measures {

struct trip_recommendation {
  recipients recipients_;
  time time_{};
  std::vector<extern_trip> planned_trips_;
  std::vector<std::uint32_t> planned_destinations_;
  extern_trip recommended_trip_{};
};

}  // namespace motis::paxforecast::measures
