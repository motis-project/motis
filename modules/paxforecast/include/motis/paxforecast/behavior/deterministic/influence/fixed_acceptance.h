#pragma once

#include <limits>

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::deterministic::influence {

struct fixed_acceptance {
  explicit fixed_acceptance(float acceptance_rate)
      : acceptance_rate_(acceptance_rate) {}

  void update_probabilities(
      motis::paxmon::passenger_group const& grp,
      std::vector<alternative> const& alternatives,
      std::vector<measures::please_use> const& announcements,
      std::vector<float>& probabilities) {
    auto const recommended = get_recommended_alternative(grp, announcements);
    if (!recommended) {
      return;
    }
    auto const recommended_index = recommended.value();
    if (probabilities[recommended_index] == 1.0F) {
      return;
    }
    auto const other_probabilities_sum =
        1.0F - probabilities[recommended_index];

    auto const other_acceptance_rate = 1.0F - acceptance_rate_;
    probabilities[recommended_index] = acceptance_rate_;
    for (auto i = 0ULL; i < alternatives.size(); ++i) {
      if (i != recommended_index) {
        probabilities[i] =
            probabilities[i] / other_probabilities_sum * other_acceptance_rate;
      }
    }
  }

  float acceptance_rate_;
};

}  // namespace motis::paxforecast::behavior::deterministic::influence
