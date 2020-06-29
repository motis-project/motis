#pragma once

#include <numeric>

#include "utl/enumerate.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::distribution {

struct partial {
  static std::vector<double> distribute(std::uint16_t passengers,
                                        std::vector<double> const& alt_scores) {
    auto result = std::vector<double>(alt_scores.size());

    auto const score_sum =
        std::accumulate(begin(alt_scores), end(alt_scores), 0.0);

    for (auto const [i, score] : utl::enumerate(alt_scores)) {
      result[i] = (score / score_sum) * passengers;
    }

    return result;
  }
};

}  // namespace motis::paxforecast::behavior::distribution
