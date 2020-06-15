#pragma once

#include <limits>

#include "utl/enumerate.h"

#include "motis/rsl/alternatives.h"
#include "motis/rsl/behavior/util.h"
#include "motis/rsl/measures/measures.h"
#include "motis/rsl/passenger_group.h"

namespace motis::rsl::behavior::distribution {

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

}  // namespace motis::rsl::behavior::distribution
