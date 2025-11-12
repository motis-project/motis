#pragma once

#include <filesystem>

#include "geo/latlng.h"

struct tg_geom;

namespace motis::odm {

struct bounds {
  explicit bounds(std::filesystem::path const&);
  ~bounds();

  bool contains(geo::latlng const&) const;

  tg_geom* geom_{nullptr};
};

struct taxi_bounds : bounds {
  explicit taxi_bounds(std::filesystem::path const&);
};

struct ride_sharing_bounds : bounds {
  explicit ride_sharing_bounds(std::filesystem::path const&);
};

}  // namespace motis::odm