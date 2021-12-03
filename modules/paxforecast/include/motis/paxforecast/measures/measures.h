#pragma once

#include <map>
#include <variant>

#include "motis/core/schedule/time.h"
#include "motis/vector.h"

#include "motis/paxforecast/measures/trip_load_information.h"
#include "motis/paxforecast/measures/trip_recommendation.h"

namespace motis::paxforecast::measures {

using measure_variant =
    std::variant<trip_load_information, trip_recommendation>;

using measure_set = mcd::vector<measure_variant>;
using measure_collection = std::map<time, measure_set>;

}  // namespace motis::paxforecast::measures
