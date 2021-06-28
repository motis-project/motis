#pragma once

#include <vector>

#include "motis/paxmon/loader/unmatched_journey.h"

namespace motis::paxmon::loader {

struct loader_result {
  std::size_t loaded_journeys_{};
  std::vector<unmatched_journey> unmatched_journeys_;
};

}  // namespace motis::paxmon::loader
