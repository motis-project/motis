#pragma once

#include <cstdint>

#include "motis/core/schedule/time.h"

namespace motis::paxmon {

enum class transfer_direction_t : std::uint8_t { ENTER, EXIT };

struct broken_transfer_info {
  std::uint16_t leg_index_{};
  transfer_direction_t direction_{};
  time current_arrival_time_{INVALID_TIME};
  time current_departure_time_{INVALID_TIME};
  std::uint16_t required_transfer_time_{};
  bool arrival_canceled_{};
  bool departure_canceled_{};
};

}  // namespace motis::paxmon
