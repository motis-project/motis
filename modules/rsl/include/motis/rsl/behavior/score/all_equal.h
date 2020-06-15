#pragma once

#include "motis/rsl/alternatives.h"

namespace motis::rsl::behavior::score {

struct all_equal {
  static double get_score(alternative const& /*alt*/) { return 1.0; }
};

}  // namespace motis::rsl::behavior::score
