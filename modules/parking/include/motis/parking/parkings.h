#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "geo/point_rtree.h"

#include "motis/parking/parking_lot.h"

namespace motis::parking {

struct parkings {
  parkings() = default;
  explicit parkings(std::string const& filename);
  explicit parkings(std::vector<parking_lot>&& parkings);

  std::vector<parking_lot> get_parkings(geo::latlng const& center,
                                        double radius) const;
  std::optional<parking_lot> get_parking(int32_t id) const;

  std::vector<parking_lot> parkings_;

private:
  geo::point_rtree rtree_;
};

}  // namespace motis::parking
