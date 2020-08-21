#pragma once

#include <cstdint>

#include "motis/core/schedule/time.h"

namespace motis::paxmon::loader {

struct unmatched_journey {
  std::uint32_t start_station_idx_{};
  std::uint32_t destination_station_idx_{};
  time departure_time_{};
};

}  // namespace motis::paxmon::loader
