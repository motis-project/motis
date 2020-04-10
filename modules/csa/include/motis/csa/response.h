#pragma once

#include <vector>

#include "motis/core/schedule/interval.h"
#include "motis/core/journey/journey.h"

#include "motis/csa/csa_journey.h"
#include "motis/csa/csa_statistics.h"

namespace motis::csa {

struct response {
  csa_statistics stats_;
  std::vector<csa_journey> journeys_;
  interval searched_interval_;
};

}  // namespace motis::csa
