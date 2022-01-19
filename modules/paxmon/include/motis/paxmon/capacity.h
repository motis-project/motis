#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "cista/reflection/comparable.h"

#include "motis/data.h"
#include "motis/hash_map.h"

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/capacity_data.h"

namespace motis::paxmon {

struct cap_trip_id {
  CISTA_COMPARABLE()

  std::uint32_t train_nr_{};
  std::uint32_t from_station_idx_{};
  std::uint32_t to_station_idx_{};
  time departure_{};
  time arrival_{};
};

using trip_capacity_map_t = std::map<cap_trip_id, std::uint16_t>;
using category_capacity_map_t = mcd::hash_map<mcd::string, std::uint16_t>;

std::size_t load_capacities(schedule const& sched,
                            std::string const& capacity_file,
                            trip_capacity_map_t& trip_map,
                            category_capacity_map_t& category_map,
                            std::string const& match_log_file = "");

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    trip_capacity_map_t const& trip_map,
    category_capacity_map_t const& category_map);

}  // namespace motis::paxmon
