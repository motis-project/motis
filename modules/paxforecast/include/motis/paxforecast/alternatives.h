#pragma once

#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reachability.h"

#include "motis/paxforecast/routing_cache.h"

namespace motis::paxforecast {

struct alternative {
  journey journey_;
  motis::paxmon::compact_journey compact_journey_;
  time arrival_time_{INVALID_TIME};
  duration duration_{};
  unsigned transfers_{};
};

std::vector<alternative> find_alternatives(
    schedule const& sched, routing_cache& cache,
    unsigned destination_station_id,
    motis::paxmon::passenger_localization const& localization,
    bool use_cache = true, duration pretrip_interval_length = 0);

}  // namespace motis::paxforecast
