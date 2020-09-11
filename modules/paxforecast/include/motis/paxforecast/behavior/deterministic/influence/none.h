#pragma once

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::deterministic::influence {

struct none {
  inline void update_probabilities(
      motis::paxmon::passenger_group const& /*grp*/,
      std::vector<alternative> const& /*alternatives*/,
      std::vector<measures::please_use> const& /*announcements*/,
      std::vector<float>& /*probabilities*/) {}
};

}  // namespace motis::paxforecast::behavior::deterministic::influence
