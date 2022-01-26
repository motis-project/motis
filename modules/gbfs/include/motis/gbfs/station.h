#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "cista/reflection/comparable.h"

#include "geo/latlng.h"

namespace motis::gbfs {

struct station {
  CISTA_COMPARABLE()
  std::string id_;
  std::string name_;
  geo::latlng pos_;
};

std::vector<station> parse_stations(std::string_view json);

}  // namespace motis::gbfs
