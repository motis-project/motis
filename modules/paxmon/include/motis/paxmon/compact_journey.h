#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon {

struct journey_leg {
  CISTA_COMPARABLE()

  trip const* trip_{nullptr};
  unsigned enter_station_id_{0};
  unsigned exit_station_id_{0};
  motis::time enter_time_{0};
  motis::time exit_time_{0};
  std::optional<transfer_info> enter_transfer_;
};

struct compact_journey {
  CISTA_COMPARABLE()

  inline unsigned destination_station_id() const {
    return legs_.back().exit_station_id_;
  }

  inline duration scheduled_duration() const {
    return !legs_.empty() ? legs_.back().exit_time_ - legs_.front().enter_time_
                          : 0;
  }

  inline time scheduled_departure_time() const {
    return !legs_.empty() ? legs_.front().enter_time_ : INVALID_TIME;
  }

  inline time scheduled_arrival_time() const {
    return !legs_.empty() ? legs_.back().exit_time_ : INVALID_TIME;
  }

  std::vector<journey_leg> legs_;
};

}  // namespace motis::paxmon
