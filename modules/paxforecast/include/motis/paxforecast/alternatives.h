#pragma once

#include <vector>

#include "motis/hash_map.h"
#include "motis/pair.h"
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

  // set by simulation:
  float pick_probability_{};
};

struct alternative_routing_options {
  bool use_cache_{};
  duration pretrip_interval_length_{};
  bool allow_start_metas_{};
  bool allow_dest_metas_{};
};

struct alternatives_request {
  motis::paxmon::passenger_localization localization_{};
  unsigned destination_station_id_{};

  // result, set by alternative routing:
  std::vector<alternative> alternatives_;
};

struct alternatives_set {
  std::vector<alternatives_request> requests_;
  mcd::hash_map<mcd::pair<motis::paxmon::passenger_localization,
                          unsigned /* destination_station_id */>,
                std::uint32_t>
      request_key_to_idx_;

  std::uint32_t add_request(
      motis::paxmon::passenger_localization const& localization,
      unsigned destination_station_id);

  void find(motis::paxmon::universe const& uv, schedule const& sched,
            routing_cache& cache, alternative_routing_options const& options);
};

std::vector<alternative> find_alternatives(
    motis::paxmon::universe const& uv, schedule const& sched,
    routing_cache& cache, unsigned const destination_station_id,
    motis::paxmon::passenger_localization const& localization,
    alternative_routing_options options);

}  // namespace motis::paxforecast
