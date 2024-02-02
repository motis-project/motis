#pragma once

#include <random>

#include "motis/paxforecast/behavior/probabilistic/passenger_behavior.h"

namespace motis::paxforecast::behavior {

inline probabilistic::passenger_behavior get_default_behavior(bool best_only = false,
                            unsigned sample_count = 1000) {
  return probabilistic::passenger_behavior{
    sample_count, best_only,
    std::normal_distribution{30.0F, 10.0F},
    std::normal_distribution{-5.0F, 3.0F},
    std::normal_distribution{-120.0F, 5.0F},
    std::normal_distribution{1.0F, 0.3F},
    std::normal_distribution{-2.0F, 3.0F},
    std::normal_distribution{15.0F, 5.0F},
    std::normal_distribution{60.0F, 10.0F},
    std::normal_distribution{0.0F, 5.0F}
  };
}

}  // namespace motis::paxforecast::behavior
