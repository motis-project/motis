#pragma once

#include <limits>

#include "utl/enumerate.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::distribution {

struct best_only {
  static std::vector<double> distribute(std::uint16_t passengers,
                                        std::vector<double> const& alt_scores) {
    auto result = std::vector<double>(alt_scores.size());

    auto selected = min_indices(alt_scores);
    auto const equal = selected.size();
    auto const per_alternative = static_cast<double>(passengers) / equal;
    for (auto i : selected) {
      result[i] = per_alternative;
    }

    return result;
  }
};

}  // namespace motis::paxforecast::behavior::distribution
