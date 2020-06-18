#pragma once

#include <cstdint>

#include "motis/rsl/passenger_group.h"
#include "motis/rsl/reachability.h"

namespace motis::rsl {

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
};

}  // namespace motis::rsl
