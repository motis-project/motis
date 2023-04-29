#include <cstdio>
#include <array>

#include "motis/parking/parkings.h"

#include "utl/pipes.h"

namespace motis::parking {

parkings::parkings(database& db) : db_{db} {
  parkings_ = db_.get_parking_lots();
  for (auto const& lot : parkings_) {
    if (lot.is_from_parkendd()) {
      auto const& info = std::get<parkendd_parking_lot_info>(lot.info_);
      parkendd_id_to_parking_lot_id_[info.id_] = lot.id_;
    }
  }
  create_rtree();
}

void parkings::create_rtree() {
  rtree_ = geo::make_point_rtree(parkings_,
                                 [](auto const& p) { return p.location_; });
}

std::vector<parking_lot> parkings::get_parkings(geo::latlng const& center,
                                                double radius) {
  std::lock_guard const lock{mutex_};
  return utl::all(rtree_.in_radius(center, radius))  //
         | utl::transform(
               [this](std::size_t index) { return parkings_[index]; })  //
         | utl::remove_if([&](auto const& lot) {
             return unavailable_parking_lots_.find(lot.id_) !=
                    end(unavailable_parking_lots_);
           })  //
         | utl::vec();
}

std::optional<parking_lot> parkings::get_parking(int32_t id) {
  std::lock_guard const lock{mutex_};
  if (id > 0 && static_cast<std::size_t>(id) <= parkings_.size()) {
    return parkings_[id - 1];
  } else {
    return {};
  }
}

void parkings::add_parkings(std::vector<parking_lot> const& parking_lots) {
  std::lock_guard const lock{mutex_};
  for (auto const& lot : parking_lots) {
    parkings_.emplace_back(lot);
    if (lot.is_from_parkendd()) {
      auto const& info = std::get<parkendd_parking_lot_info>(lot.info_);
      parkendd_id_to_parking_lot_id_[info.id_] = lot.id_;
    }
  }
  create_rtree();
}

std::int32_t parkings::get_parkendd_lot_id(std::string_view parkendd_id) {
  std::lock_guard const lock{mutex_};
  return parkendd_id_to_parking_lot_id_.at(parkendd_id);
}

void parkings::set_unavailable_parking_lots(
    mcd::hash_set<std::int32_t>&& unavailable) {
  std::lock_guard const lock{mutex_};
  unavailable_parking_lots_ = std::move(unavailable);
}

}  // namespace motis::parking
