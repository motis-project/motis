#pragma once

#include "motis/paxforecast/alternatives.h"

namespace motis::paxforecast::behavior::deterministic::score {

struct all_equal {
  static double get_score(alternative const& /*alt*/) { return 1.0; }
};

}  // namespace motis::paxforecast::behavior::deterministic::score
