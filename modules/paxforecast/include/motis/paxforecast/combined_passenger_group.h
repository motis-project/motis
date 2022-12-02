#pragma once

#include <cstdint>
#include <vector>

#include "motis/core/journey/journey.h"

#include "motis/paxmon/index_types.h"
#include "motis/paxmon/localization.h"

#include "motis/paxforecast/alternatives.h"

namespace motis::paxforecast {

// probability is stored because it may be changed before the simulation begins
// passenger count is stored for convenience
struct passenger_group_with_route_and_probability {
  motis::paxmon::passenger_group_with_route pgwr_{};
  float probability_{};
  std::uint16_t passengers_{};
};

struct combined_passenger_group {
  unsigned destination_station_id_{};
  std::uint16_t passengers_{};
  bool has_major_delay_groups_{false};
  motis::paxmon::passenger_localization localization_;
  std::vector<passenger_group_with_route_and_probability> group_routes_;
  std::vector<alternative> alternatives_;
};

}  // namespace motis::paxforecast
