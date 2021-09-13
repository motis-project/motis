#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/journey/extern_trip.h"

#include "motis/paxforecast/measures/interval.h"
#include "motis/paxforecast/measures/recipients.h"

namespace motis::paxforecast::measures {

struct trip_recommendation {
  recipients recipients_;
  interval interval_;
  std::vector<extern_trip> planned_trips_;
  std::vector<std::uint32_t> planned_destinations_;
  extern_trip recommended_trip_{};
  std::uint32_t interchange_station_{};
};

}  // namespace motis::paxforecast::measures
