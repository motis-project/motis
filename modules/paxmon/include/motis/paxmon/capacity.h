#pragma once

#include <cstdint>
#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

struct detailed_capacity {
  std::uint16_t seats_{};
  std::uint16_t seats_1st_{};
  std::uint16_t seats_2nd_{};
  std::uint16_t standing_{};
  std::uint16_t total_limit_{};

  detailed_capacity& operator+=(detailed_capacity const& o) {
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
  detailed_capacity total_capacity_{};
};

struct capacity_maps {
  std::map<cap_trip_id, std::uint16_t> trip_capacity_map_;
  mcd::hash_map<mcd::string, std::uint16_t> category_capacity_map_;

  mcd::hash_map<std::uint64_t /* UIC number */, detailed_capacity>
      vehicle_capacity_map_;
  mcd::hash_map<boost::uuids::uuid, trip_formation> trip_formation_map_;
  mcd::hash_map<primary_trip_id, boost::uuids::uuid> trip_uuid_map_;
  mcd::hash_map<boost::uuids::uuid, primary_trip_id> uuid_trip_map_;

  mcd::hash_map<mcd::string, detailed_capacity> vehicle_group_capacity_map_;
  mcd::hash_map<mcd::string, detailed_capacity> gattung_capacity_map_;
  mcd::hash_map<mcd::string, detailed_capacity> baureihe_capacity_map_;

  mcd::hash_map<cap_trip_id, mcd::vector<capacity_override_section>>
      override_map_;

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

struct vehicle_capacity {
  vehicle_info const* vehicle_{};
  detailed_capacity capacity_{};
  capacity_source source_{capacity_source::UNKNOWN};
  bool duplicate_vehicle_{};
};

struct vehicle_group_capacity {
  vehicle_group const* group_{};
  trip const* trp_{};
  detailed_capacity capacity_{};
  capacity_source source_{capacity_source::UNKNOWN};
  bool duplicate_group_{};
  std::vector<vehicle_capacity> vehicles_;
};

struct trip_capacity {
  inline bool has_trip_lookup_capacity() const {
    return trip_lookup_source_ != capacity_source::UNKNOWN;
  }

  inline bool has_formation() const {
    return formation_ != nullptr && formation_section_ != nullptr;
  }

  inline bool has_formation_capacity() const {
    return formation_source_ != capacity_source::UNKNOWN;
  }

  trip const* trp_{};
  trip_formation const* formation_{};
  trip_formation_section const* formation_section_{};
  connection const* full_con_{};
  connection_info const* con_info_{};

  detailed_capacity trip_lookup_capacity_{};
  capacity_source trip_lookup_source_{capacity_source::UNKNOWN};

  detailed_capacity formation_capacity_{};
  capacity_source formation_source_{capacity_source::UNKNOWN};
};

struct section_capacity {
  inline bool has_capacity() const {
    return source_ != capacity_source::UNKNOWN;
  }

  inline bool is_overridden() const {
    return source_ == capacity_source::OVERRIDE;
  }

  inline detailed_capacity const& original_capacity() const {
    return is_overridden() ? original_capacity_ : capacity_;
  }

  inline capacity_source original_capacity_source() const {
    return is_overridden() ? original_source_ : source_;
  }

  detailed_capacity capacity_{};
  capacity_source source_{capacity_source::UNKNOWN};

  // only set if is_overridden()
  detailed_capacity original_capacity_{};
  capacity_source original_source_{capacity_source::UNKNOWN};

  // the following fields are only set if detailed = true is set
  // during the lookup (-> get_capacity)

  std::uint16_t num_vehicles_{};
  std::uint16_t num_vehicles_uic_found_{};
  std::uint16_t num_vehicles_baureihe_used_{};
  std::uint16_t num_vehicles_gattung_used_{};
  std::uint16_t num_vehicles_no_info_{};
  std::uint16_t num_vehicle_groups_used_{};

  std::vector<trip_capacity> trips_;
  std::vector<vehicle_group_capacity> vehicle_groups_;
};

section_capacity get_capacity(schedule const& sched, light_connection const& lc,
                              ev_key const& ev_key_from,
                              ev_key const& ev_key_to,
                              capacity_maps const& caps, bool detailed = false);

}  // namespace motis::paxmon
