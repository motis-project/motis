#pragma once

#include <vector>

#include "motis/core/journey/journey.h"

#include "motis/rsl/alternatives.h"
#include "motis/rsl/localization.h"
#include "motis/rsl/passenger_group.h"

namespace motis::rsl {

struct combined_passenger_group {
  unsigned destination_station_id_{};
  std::uint16_t passengers_{};
  passenger_localization localization_;
  std::vector<passenger_group*> groups_;
  std::vector<alternative> alternatives_;
};

}  // namespace motis::rsl
