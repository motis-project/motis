#pragma once

#include "motis/rsl/alternatives.h"
#include "motis/rsl/behavior/util.h"
#include "motis/rsl/measures/measures.h"
#include "motis/rsl/passenger_group.h"

namespace motis::rsl::behavior::influence {

struct multiplicative {
  void update_scores(passenger_group const& grp,
                     std::vector<alternative> const& alternatives,
                     std::vector<measures::please_use> const& announcements,
                     std::vector<double>& scores) {
    auto const recommended = get_recommended_alternative(grp, announcements);
    if (recommended) {
      auto const recommended_index = recommended.value();
      for (auto i = 0ULL; i < alternatives.size(); ++i) {
        if (i == recommended_index) {
          scores[i] *= recommended_factor_;
        } else {
          scores[i] *= not_recommended_factor_;
        }
      }
    }
  }

  double recommended_factor_{1.0};
  double not_recommended_factor_{1.0};
};

}  // namespace motis::rsl::behavior::influence
