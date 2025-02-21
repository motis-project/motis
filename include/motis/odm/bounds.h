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

}  // namespace motis::odm