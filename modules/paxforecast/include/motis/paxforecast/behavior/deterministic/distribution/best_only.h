#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

namespace motis::paxforecast::behavior::deterministic::distribution {

struct best_only {
  static std::vector<float> get_probabilities(
      std::vector<double> const& scores) {
    std::vector<float> result;
    result.resize(scores.size());
    result[std::distance(begin(scores),
                         std::min_element(begin(scores), end(scores)))] = 1.0F;
    return result;
  }
};

}  // namespace motis::paxforecast::behavior::deterministic::distribution
