#pragma once

#include "motis/paxforecast/behavior/probabilistic/passenger_behavior.h"

namespace motis::paxforecast::behavior {

probabilistic::passenger_behavior parse_passenger_behavior(
    std::string const& json, unsigned default_sample_count = 1000,
    bool default_best_only = false);

}  // namespace motis::paxforecast::behavior
