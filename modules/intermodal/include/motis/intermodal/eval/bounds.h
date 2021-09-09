#pragma once

#include "geo/detail/register_latlng.h"
#include "geo/latlng.h"

namespace motis::intermodal::eval {

class bounds {
public:
  virtual ~bounds() = default;
  bounds() = default;
  bounds(bounds const&) = delete;
  bounds& operator=(bounds const&) = delete;
  bounds(bounds&&) = delete;
  bounds& operator=(bounds&&) = delete;

  virtual bool contains(geo::latlng const& loc) const = 0;
  virtual geo::latlng random_pt() = 0;
};

}  // namespace motis::intermodal::eval
