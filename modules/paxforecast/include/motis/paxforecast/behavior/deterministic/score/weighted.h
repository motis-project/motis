#pragma once

#include "motis/paxforecast/alternatives.h"

namespace motis::paxforecast::behavior::deterministic::score {

struct weighted {
  double get_score(alternative const& alt) const {
    return duration_ * alt.duration_ + transfers_ * alt.transfers_;
  }

  double duration_{1.0};
  double transfers_{5.0};
};

}  // namespace motis::paxforecast::behavior::deterministic::score
