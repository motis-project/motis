#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "geo/point_rtree.h"

#include "motis/parking/database.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

struct parkings {
  explicit parkings(database& db);

  std::vector<parking_lot> get_parkings(geo::latlng const& center,
                                        double radius) const;
  std::optional<parking_lot> get_parking(int32_t id) const;

  std::vector<parking_lot> parkings_;

private:
  geo::point_rtree rtree_;
  database& db_;
};

}  // namespace motis::parking
