#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "motis/core/schedule/time.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon {

struct journey_leg {
  motis::extern_trip trip_;
  unsigned enter_station_id_;
  unsigned exit_station_id_;
  motis::time enter_time_;
  motis::time exit_time_;
  std::optional<transfer_info> enter_transfer_;
};

struct compact_journey {
  std::vector<journey_leg> legs_;

  inline unsigned destination_station_id() const {
    return legs_.back().exit_station_id_;
  }
};

}  // namespace motis::paxmon
