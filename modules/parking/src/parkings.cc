#include <cstdio>
#include <array>

#include "motis/parking/parkings.h"

#include "motis/core/common/logging.h"

#include "utl/to_vec.h"

using namespace motis::logging;

namespace motis::parking {

parkings::parkings(database& db) : db_{db} {
  parkings_ = db_.get_parking_lots();
  LOG(info) << "parking::parkings(): loaded " << parkings_.size()
            << " parkings from db";
  rtree_ = geo::make_point_rtree(parkings_,
                                 [](auto const& p) { return p.location_; });
}

std::vector<parking_lot> parkings::get_parkings(geo::latlng const& center,
                                                double radius) const {
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

}  // namespace motis::parking
