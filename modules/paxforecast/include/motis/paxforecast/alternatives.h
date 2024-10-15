#pragma once

#include <vector>

#include "motis/vector.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/universe.h"

#include "motis/paxforecast/routing_cache.h"

#include "motis/paxforecast/measures/load_level.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast {

struct alternative {
  journey journey_;
  motis::paxmon::compact_journey compact_journey_;
  time arrival_time_{INVALID_TIME};
  duration duration_{};
  unsigned transfers_{};
  bool is_original_{};
  bool is_recommended_{};
  measures::load_level load_info_{measures::load_level::UNKNOWN};
};

std::vector<alternative> find_alternatives(
    motis::paxmon::universe const& uv, schedule const& sched,
    routing_cache& cache,
    mcd::vector<measures::measure_variant const*> const& group_measures,
    unsigned const destination_station_id,
    motis::paxmon::passenger_localization const& localization,
    motis::paxmon::compact_journey const* remaining_journey, bool use_cache,
    duration pretrip_interval_length);

}  // namespace motis::paxforecast
