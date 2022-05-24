#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "boost/uuid/uuid.hpp"

#include "cista/reflection/comparable.h"

#include "motis/data.h"
#include "motis/hash_map.h"

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/event.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/capacity_data.h"
#include "motis/paxmon/vehicle_order.h"

namespace motis::paxmon {

struct cap_trip_id {
  CISTA_COMPARABLE()

  std::uint32_t train_nr_{};
  std::uint32_t from_station_idx_{};
  std::uint32_t to_station_idx_{};
  time departure_{};
  time arrival_{};
};

struct vehicle_capacity {
  std::uint16_t seats_{};
  std::uint16_t standing_{};
  std::uint16_t total_limit_{};

  inline std::uint16_t limit() const {
    return total_limit_ != 0U ? total_limit_ : seats_ + standing_;
  }
};

using trip_capacity_map_t = std::map<cap_trip_id, std::uint16_t>;
using category_capacity_map_t = mcd::hash_map<mcd::string, std::uint16_t>;
using vehicle_capacity_map_t =
    mcd::hash_map<std::uint64_t /* UIC number */, vehicle_capacity>;
using trip_vehicle_map_t =
    mcd::hash_map<boost::uuids::uuid, trip_vehicle_order>;

struct capacity_maps {
  trip_capacity_map_t trip_capacity_map_;
  category_capacity_map_t category_capacity_map_;
  vehicle_capacity_map_t vehicle_capacity_map_;
  trip_vehicle_map_t trip_vehicle_map_;
};

std::size_t load_capacities(schedule const& sched,
                            std::string const& capacity_file,
                            capacity_maps& caps,
                            std::string const& match_log_file = "");

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    ev_key const& ev_key_from, ev_key const& ev_key_to,
    capacity_maps const& caps);

}  // namespace motis::paxmon
