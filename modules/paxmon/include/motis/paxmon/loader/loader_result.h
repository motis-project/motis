#pragma once

#include <vector>
#include <cstddef>

#include "motis/paxmon/loader/unmatched_journey.h"

namespace motis::paxmon::loader {

struct loader_result {
  std::vector<unmatched_journey> unmatched_journeys_;

  std::size_t loaded_journey_count_{};
  std::size_t loaded_group_count_{};
  std::size_t loaded_pax_count_{};

  std::size_t unmatched_journey_count_{};
  std::size_t unmatched_group_count_{};
  std::size_t unmatched_pax_count_{};
};

}  // namespace motis::paxmon::loader
