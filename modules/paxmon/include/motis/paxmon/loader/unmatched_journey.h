#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/schedule/time.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon::loader {

struct unmatched_journey {
  std::uint32_t start_station_idx_{};
  std::uint32_t destination_station_idx_{};
  time departure_time_{INVALID_TIME};
  time arrival_time_{INVALID_TIME};
  motis::paxmon::data_source source_{};
  std::uint16_t passengers_{};
  std::vector<std::uint16_t> group_sizes_;
};

}  // namespace motis::paxmon::loader
