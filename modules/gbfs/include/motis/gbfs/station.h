#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/hash_map.h"
#include "geo/latlng.h"

namespace motis::gbfs {

struct station {
  CISTA_COMPARABLE()
  std::string id_;
  std::string name_;
  geo::latlng pos_;
  unsigned bikes_available_{0};
  mcd::hash_map<std::string, unsigned> vehicles_available_{};
};

mcd::hash_map<std::string, station> parse_stations(std::string const& tag,
                                                   std::string_view info,
                                                   std::string_view status);

}  // namespace motis::gbfs
