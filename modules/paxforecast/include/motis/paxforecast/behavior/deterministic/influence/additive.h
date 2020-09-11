#pragma once

#include <algorithm>

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::deterministic::influence {

struct additive {
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
    auto const old_probability = probabilities[recommended_index];
    if (old_probability == 1.0F) {
      return;
    }
    auto const new_probability =
        std::min(1.0F, old_probability + recommended_boost_);
    probabilities[recommended_index] = new_probability;
    auto const other_factor =
        (1.0F - new_probability) / (1.0F - old_probability);
    for (auto i = 0ULL; i < alternatives.size(); ++i) {
      if (i != recommended_index) {
        probabilities[i] *= other_factor;
      }
    }
  }

  float recommended_boost_{0.0};
};

}  // namespace motis::paxforecast::behavior::deterministic::influence
