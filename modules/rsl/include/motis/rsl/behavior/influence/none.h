#pragma once

#include "motis/rsl/alternatives.h"
#include "motis/rsl/behavior/util.h"
#include "motis/rsl/measures/measures.h"
#include "motis/rsl/passenger_group.h"

namespace motis::rsl::behavior::influence {

struct none {
  inline void update_scores(
      passenger_group const& /*grp*/,
      std::vector<alternative> const& /*alternatives*/,
      std::vector<measures::please_use> const& /*announcements*/,
      std::vector<double>& /*scores*/) {}
};

}  // namespace motis::rsl::behavior::influence
