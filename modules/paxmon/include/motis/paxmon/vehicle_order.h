#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "boost/uuid/uuid.hpp"

#include "cista/reflection/comparable.h"

#include "motis/hash_map.h"
#include "motis/string.h"

namespace motis::paxmon {

struct station_range {
  CISTA_COMPARABLE()

  mcd::string from_eva_;
  mcd::string to_eva_;
};

struct vehicle_order {
  std::vector<std::uint64_t> uics_;
};

struct section_vehicle_order {
  mcd::string departure_eva_;
  boost::uuids::uuid departure_uuid_{};
  vehicle_order vehicles_;
};

struct trip_vehicle_order {
  mcd::hash_map<station_range, vehicle_order> station_ranges_;
  std::vector<section_vehicle_order> sections_;
};

}  // namespace motis::paxmon
