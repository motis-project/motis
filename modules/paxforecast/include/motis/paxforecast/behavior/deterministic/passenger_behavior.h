#pragma once

#include <vector>

#include "utl/to_vec.h"

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::deterministic {

template <typename Score, typename Distribution, typename Influence>
struct passenger_behavior {
  passenger_behavior(Score&& score, Distribution&& distribution,
                     Influence&& influence)
      : score_(std::move(score)),
        distribution_(std::move(distribution)),
        influence_(std::move(influence)) {}

  std::vector<float> pick_routes(
      motis::paxmon::passenger_group const& grp,
      std::vector<alternative> const& alternatives,
      std::vector<measures::please_use> const& announcements) {
    if (alternatives.empty()) {
      return {};
    }
    auto const scores = utl::to_vec(alternatives, [&](alternative const& alt) {
      return score_.get_score(alt);
    });
    auto probabilities = distribution_.get_probabilities(scores);
    influence_.update_probabilities(grp, alternatives, announcements,
                                    probabilities);
    return probabilities;
  }

  Score score_;
  Distribution distribution_;
  Influence influence_;
};

}  // namespace motis::paxforecast::behavior::deterministic
