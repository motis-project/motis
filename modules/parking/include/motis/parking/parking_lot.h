#pragma once

#include <cstdint>

#include "geo/latlng.h"

namespace motis::parking {

struct parking_lot {
  parking_lot() = default;
  parking_lot(int32_t id, geo::latlng location, bool fee)
      : id_(id), location_(location), fee_(fee) {}

  bool valid() const { return id_ != 0; }

  int32_t id_{0};
  geo::latlng location_;
  bool fee_{false};
};

}  // namespace motis::parking
