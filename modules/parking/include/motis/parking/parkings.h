#pragma once

#include <cstdint>
#include <mutex>
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
                                        double radius);
  std::optional<parking_lot> get_parking(int32_t id) const;

  void add_parkings(std::vector<parking_lot> const& parking_lots);

  std::vector<parking_lot> parkings_;

private:
  void create_rtree();

  std::mutex rtree_mutex_;
  geo::point_rtree rtree_;
  database& db_;
};

}  // namespace motis::parking
