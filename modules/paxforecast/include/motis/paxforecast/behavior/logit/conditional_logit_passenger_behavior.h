#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"

namespace motis::paxforecast::behavior::logit {

struct conditional_logit_passenger_behavior {
  conditional_logit_passenger_behavior(float coeff_duration,
                                       float coeff_transfer, bool best_only)
      : coeff_duration_{coeff_duration},
        coeff_transfer_{coeff_transfer},
        best_only_{best_only} {}

  std::vector<float> pick_routes(std::vector<alternative> const& alternatives) {
    if (alternatives.empty()) {
      return {};
    }

    auto const denominator =
        std::accumulate(begin(alternatives), end(alternatives), 0.0F,
                        [&](float const sum, alternative const& alt) {
                          return sum + std::exp(utility(alt));
                        });
    auto probabilities = utl::to_vec(alternatives, [&](alternative const& alt) {
      return std::exp(utility(alt)) / denominator;
    });
    if (best_only_) {
      only_keep_best_alternative(probabilities);
    }

    auto const prob_sum =
        std::accumulate(begin(probabilities), end(probabilities), 0.0F);
    utl::verify(prob_sum >= 0.99F && prob_sum <= 1.01F, "probability sum = {}",
                prob_sum);

    return probabilities;
  }

private:
  inline float utility(alternative const& alt) const {
    return coeff_duration_ * (-static_cast<float>(alt.duration_)) +
           coeff_transfer_ * (-static_cast<float>(alt.transfers_));
  }

  float coeff_duration_{};
  float coeff_transfer_{};
  bool best_only_{false};
};

}  // namespace motis::paxforecast::behavior::logit
