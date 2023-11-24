#pragma once

#include <vector>
#include <cstddef>

#include "motis/paxmon/loader/unmatched_journey.h"

namespace motis::paxmon::loader {

struct loader_result {
  std::size_t loaded_journeys_{};
  std::vector<unmatched_journey> unmatched_journeys_;

  std::size_t loaded_pax_{};
  std::size_t unmatched_pax_{};
};

}  // namespace motis::paxmon::loader
