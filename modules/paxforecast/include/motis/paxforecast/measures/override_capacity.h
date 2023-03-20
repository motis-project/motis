#pragma once

#include <cstdint>

#include "motis/vector.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/capacity.h"

namespace motis::paxforecast::measures {

struct override_capacity {
  time time_{};
  full_trip_id trip_id_{};
  mcd::vector<motis::paxmon::capacity_override_section> sections_;
};

}  // namespace motis::paxforecast::measures
