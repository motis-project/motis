#pragma once

#include <vector>

#include "motis/paxforecast/measures/announcements.h"

#include "motis/paxforecast/measures/trip_load_information.h"
#include "motis/paxforecast/measures/trip_recommendation.h"

namespace motis::paxforecast::measures {

struct measures {
  std::vector<trip_recommendation> recommendations_;
  std::vector<trip_load_information> load_infos_;
};

}  // namespace motis::paxforecast::measures
