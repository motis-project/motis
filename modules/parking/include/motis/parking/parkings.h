#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "geo/point_rtree.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/parking/database.h"
#include "motis/parking/parking_lot.h"

namespace motis::parking {

struct parkings {
  explicit parkings(database& db);

  std::vector<parking_lot> get_parkings(geo::latlng const& center,
                                        double radius);
  std::optional<parking_lot> get_parking(int32_t id);

  void add_parkings(std::vector<parking_lot> const& parking_lots);

  std::int32_t get_parkendd_lot_id(std::string_view parkendd_id);
  void set_unavailable_parking_lots(mcd::hash_set<std::int32_t>&& unavailable);

private:
  void create_rtree();

  std::mutex mutex_;
  geo::point_rtree rtree_;
  database& db_;
  std::vector<parking_lot> parkings_;
  mcd::hash_map<std::string, std::int32_t> parkendd_id_to_parking_lot_id_;
  mcd::hash_set<std::int32_t> unavailable_parking_lots_;
};

}  // namespace motis::parking
