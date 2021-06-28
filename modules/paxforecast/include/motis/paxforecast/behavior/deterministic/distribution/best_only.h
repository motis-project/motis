#pragma once

#include <limits>
#include <numeric>
#include <vector>

#include "utl/enumerate.h"

namespace motis::paxforecast::behavior::deterministic::distribution {

struct best_only {
  static std::vector<float> get_probabilities(
      std::vector<double> const& scores) {
    auto best_idx = 0ULL;
    auto best_score = std::numeric_limits<double>::max();
    for (auto const [idx, score] : utl::enumerate(scores)) {
      if (score < best_score) {
        best_idx = idx;
        best_score = score;
      }
    }
    std::vector<float> result;
    result.resize(scores.size());
    result[best_idx] = 1.0F;
    return result;
  }
};

}  // namespace motis::paxforecast::behavior::deterministic::distribution
