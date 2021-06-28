#pragma once

#include <numeric>
#include <vector>

#include "utl/repeat_n.h"
#include "utl/to_vec.h"

namespace motis::paxforecast::behavior::deterministic::distribution {

struct proportional {
  static std::vector<float> get_probabilities(
      std::vector<double> const& scores) {
    auto const sum = std::accumulate(begin(scores), end(scores), 0.0);
    if (sum == 0.0) {
      return utl::repeat_n(1.0F / scores.size(), scores.size());
    }
    return utl::to_vec(scores, [&](auto const& score) {
      return static_cast<float>(score / sum);
    });
  }
};

}  // namespace motis::paxforecast::behavior::deterministic::distribution
