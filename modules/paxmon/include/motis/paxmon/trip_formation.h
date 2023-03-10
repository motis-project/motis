#pragma once

#include <cstdint>

#include "boost/uuid/uuid.hpp"

#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/time.h"

namespace motis::paxmon {

struct vehicle_info {
  std::uint64_t uic_{};
  mcd::string baureihe_;
  mcd::string type_code_;
  mcd::string order_;
};

struct trip_formation_section {
  mcd::string departure_eva_;
  time schedule_departure_time_{INVALID_TIME};
  mcd::vector<vehicle_info> vehicles_;
};

struct trip_formation {
  mcd::vector<trip_formation_section> sections_;
};

}  // namespace motis::paxmon
