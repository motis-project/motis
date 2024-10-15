#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/time.h"

#include "motis/paxforecast/measures/recipients.h"
#include "motis/paxforecast/measures/trip_with_load_level.h"

namespace motis::paxforecast::measures {

struct trip_load_recommendation {
  recipients recipients_;
  time time_{};
  std::vector<std::uint32_t> planned_destinations_;
  std::vector<trip_with_load_level> full_trips_;
  std::vector<trip_with_load_level> recommended_trips_;
};

}  // namespace motis::paxforecast::measures
