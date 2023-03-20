#pragma once

#include <cstdint>
#include <algorithm>
#include <map>
#include <optional>
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
#include "motis/paxmon/trip_formation.h"

namespace motis::paxmon {

struct cap_trip_id {
  CISTA_COMPARABLE()

  std::uint32_t train_nr_{};
  std::uint32_t from_station_idx_{};
  time departure_{};
  std::uint32_t to_station_idx_{};
  time arrival_{};
};

struct vehicle_capacity {
  std::uint16_t seats_{};
  std::uint16_t seats_1st_{};
  std::uint16_t seats_2nd_{};
  std::uint16_t standing_{};
  std::uint16_t total_limit_{};

  vehicle_capacity& operator+=(vehicle_capacity const& o) {
    seats_ += o.seats_;
    seats_1st_ += o.seats_1st_;
    seats_2nd_ += o.seats_2nd_;
    standing_ += o.standing_;
    total_limit_ += o.total_limit_;
    return *this;
  }

  inline std::uint16_t seats() const { return seats_; }

  inline std::uint16_t limit() const {
    return total_limit_ != 0U ? total_limit_ : seats_;
  }

  inline void update_seats() {
    seats_ =
        std::max(seats_, static_cast<std::uint16_t>(seats_1st_ + seats_2nd_));
  }
};

struct capacity_override_section {
  std::uint32_t departure_station_idx_{};
  time schedule_departure_time_{INVALID_TIME};
  vehicle_capacity total_capacity_{};
};

using trip_capacity_map_t = std::map<cap_trip_id, std::uint16_t>;
using category_capacity_map_t = mcd::hash_map<mcd::string, std::uint16_t>;
using vehicle_capacity_map_t =
    mcd::hash_map<std::uint64_t /* UIC number */, vehicle_capacity>;
using trip_formation_map_t = mcd::hash_map<boost::uuids::uuid, trip_formation>;
using trip_uuid_map_t = mcd::hash_map<primary_trip_id, boost::uuids::uuid>;
using capacity_override_map_t =
    mcd::hash_map<cap_trip_id, mcd::vector<capacity_override_section>>;

struct capacity_maps {
  trip_capacity_map_t trip_capacity_map_;
  category_capacity_map_t category_capacity_map_;

  vehicle_capacity_map_t vehicle_capacity_map_;
  trip_formation_map_t trip_formation_map_;
  trip_uuid_map_t trip_uuid_map_;

  capacity_override_map_t override_map_;

  int fuzzy_match_max_time_diff_{};  // minutes
  std::uint16_t min_capacity_{};
};

inline cap_trip_id get_cap_trip_id(full_trip_id const& id,
                                   std::optional<std::uint32_t> train_nr = {}) {
  return cap_trip_id{train_nr.value_or(id.primary_.train_nr_),
                     id.primary_.get_station_id(), id.primary_.get_time(),
                     id.secondary_.target_station_id_,
                     id.secondary_.target_time_};
}

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    ev_key const& ev_key_from, ev_key const& ev_key_to,
    capacity_maps const& caps);

}  // namespace motis::paxmon
