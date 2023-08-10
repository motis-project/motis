#pragma once

#include <cstdint>

#include "boost/uuid/uuid.hpp"

#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/journey/extern_trip.h"

namespace motis::paxmon {

struct vehicle_info {
  bool has_uic() const { return uic_ != 0; }

  std::uint64_t uic_{};
  mcd::string baureihe_;
  mcd::string type_code_;
  mcd::string order_;
};

struct vehicle_group {
  mcd::string name_;
  mcd::string start_eva_;
  mcd::string destination_eva_;
  boost::uuids::uuid trip_uuid_;
  extern_trip primary_trip_id_;
  mcd::vector<vehicle_info> vehicles_;
};

struct trip_formation_section {
  mcd::string departure_eva_;
  time schedule_departure_time_{INVALID_TIME};
  mcd::vector<vehicle_group> vehicle_groups_;
};

struct trip_formation {
  primary_trip_id ptid_{};
  boost::uuids::uuid uuid_;
  mcd::string category_;
  mcd::vector<trip_formation_section> sections_;
};

}  // namespace motis::paxmon
