#include <cstdio>
#include <array>

#include "motis/parking/parkings.h"

#include "utl/to_vec.h"

namespace motis::parking {

parkings::parkings(database& db) : db_{db} {
  parkings_ = db_.get_parking_lots();
  create_rtree();
}

void parkings::create_rtree() {
  std::lock_guard lock{rtree_mutex_};
  rtree_ = geo::make_point_rtree(parkings_,
                                 [](auto const& p) { return p.location_; });
}

std::vector<parking_lot> parkings::get_parkings(geo::latlng const& center,
                                                double radius) {
  std::lock_guard lock{rtree_mutex_};
  return utl::to_vec(rtree_.in_radius(center, radius),
                     [this](std::size_t index) { return parkings_[index]; });
}

std::optional<parking_lot> parkings::get_parking(int32_t id) const {
  if (id > 0 && static_cast<std::size_t>(id) <= parkings_.size()) {
    return parkings_[id - 1];
  } else {
    return {};
  }
}

void parkings::add_parkings(std::vector<parking_lot> const& parking_lots) {
  for (auto const& lot : parking_lots) {
    parkings_.emplace_back(lot);
  }
  create_rtree();
}

}  // namespace motis::parking
