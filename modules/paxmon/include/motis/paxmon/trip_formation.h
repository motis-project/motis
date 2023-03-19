#pragma once

#include <cstdint>

#include "boost/uuid/uuid.hpp"

#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/time.h"

namespace motis::paxmon {

struct trip_formation_section {
  mcd::string departure_eva_;
  time schedule_departure_time_{INVALID_TIME};
  mcd::vector<std::uint64_t> uics_;
};

struct trip_formation {
  mcd::vector<trip_formation_section> sections_;
};

}  // namespace motis::paxmon
