#pragma once

#include <cstdint>

#include "motis/core/schedule/time.h"

#include "motis/paxmon/localization.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reachability.h"

namespace motis::paxmon {

enum class monitoring_event_type : std::uint8_t {
  NO_PROBLEM,
  TRANSFER_BROKEN,
  MAJOR_DELAY_EXPECTED
};

struct monitoring_event {
  monitoring_event_type type_{monitoring_event_type::NO_PROBLEM};
  passenger_group const& group_;
  passenger_localization localization_;
  reachability_status reachability_status_{reachability_status::OK};
  time expected_arrival_time_{INVALID_TIME};
};

}  // namespace motis::paxmon
