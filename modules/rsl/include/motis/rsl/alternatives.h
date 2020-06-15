#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/rsl/compact_journey.h"
#include "motis/rsl/localization.h"
#include "motis/rsl/passenger_group.h"
#include "motis/rsl/reachability.h"

namespace motis::rsl {

struct alternative {
  journey journey_;
  compact_journey compact_journey_;
  time arrival_time_{INVALID_TIME};
  duration duration_{};
  unsigned transfers_{};
};

std::vector<alternative> find_alternatives(
    schedule const& sched, unsigned destination_station_id,
    passenger_localization const& localization);

}  // namespace motis::rsl
