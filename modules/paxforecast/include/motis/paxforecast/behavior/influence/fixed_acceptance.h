#pragma once

#include <limits>
#include <random>

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::influence {

struct fixed_acceptance {
  explicit fixed_acceptance(double acceptance_rate)
      : acceptance_rate_(acceptance_rate), mt_(std::random_device()()) {}

  void update_scores(motis::paxmon::passenger_group const& grp,
                     std::vector<alternative> const& alternatives,
                     std::vector<measures::please_use> const& announcements,
                     std::vector<double>& scores) {
    auto const recommended = get_recommended_alternative(grp, announcements);
    auto const accept = real_dist_(mt_) < acceptance_rate_;
    if (recommended && accept) {
      auto const recommended_index = recommended.value();
      for (auto i = 0ULL; i < alternatives.size(); ++i) {
        if (i != recommended_index) {
          scores[i] = std::numeric_limits<double>::max();
        }
      }
    }
  }

  double acceptance_rate_;
  std::mt19937 mt_;
  std::uniform_real_distribution<double> real_dist_;
};

}  // namespace motis::paxforecast::behavior::influence
