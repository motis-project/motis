#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/journey/extern_trip.h"

namespace motis::paxforecast::measures {

struct recipients {
  std::vector<extern_trip> trips_;
  std::vector<std::uint32_t> stations_;
};

}  // namespace motis::paxforecast::measures
