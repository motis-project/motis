#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "cista/reflection/comparable.h"

#include "geo/latlng.h"

namespace motis::gbfs {

struct free_bike {
  CISTA_COMPARABLE()
  std::string id_;
  geo::latlng pos_;
  std::string type_;
};

std::vector<free_bike> parse_free_bikes(std::string const& tag,
                                        std::string_view);

}  // namespace motis::gbfs
