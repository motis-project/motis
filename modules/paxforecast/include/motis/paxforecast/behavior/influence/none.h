#pragma once

#include "motis/paxmon/passenger_group.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/util.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::behavior::influence {

struct none {
  inline void update_scores(
      motis::paxmon::passenger_group const& /*grp*/,
      std::vector<alternative> const& /*alternatives*/,
      std::vector<measures::please_use> const& /*announcements*/,
      std::vector<double>& /*scores*/) {}
};

}  // namespace motis::paxforecast::behavior::influence
