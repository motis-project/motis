#pragma once

#include <vector>

#include "utl/to_vec.h"

#include "motis/rsl/alternatives.h"
#include "motis/rsl/measures/measures.h"
#include "motis/rsl/passenger_group.h"

namespace motis::rsl::behavior {

template <typename Score, typename Influence, typename Distribution,
          typename Postprocessing>
struct passenger_behavior {
  using out_assignment_t = typename Postprocessing::out_assignment_t;

  passenger_behavior(Score&& score, Influence&& influence,
                     Distribution&& distribution,
                     Postprocessing&& postprocessing)
      : score_(std::move(score)),
        influence_(std::move(influence)),
        distribution_(std::move(distribution)),
        postprocessing_(std::move(postprocessing)) {}

  std::vector<out_assignment_t> pick_routes(
      passenger_group const& grp, std::vector<alternative> const& alternatives,
      std::vector<measures::please_use> const& announcements) {
    if (alternatives.empty()) {
      return {};
    }
    auto scores = utl::to_vec(alternatives, [&](alternative const& alt) {
      return score_.get_score(alt);
    });
    influence_.update_scores(grp, alternatives, announcements, scores);
    auto const real_assignments =
        distribution_.distribute(grp.passengers_, scores);
    return postprocessing_.postprocess(real_assignments, grp.passengers_);
  }

  Score score_;
  Influence influence_;
  Distribution distribution_;
  Postprocessing postprocessing_;
};

}  // namespace motis::rsl::behavior
