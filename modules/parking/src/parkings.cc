#include <cstdio>
#include <array>
#include <string>

#include "motis/parking/parkings.h"

#include "utl/parser/cstr.h"
#include "utl/parser/csv.h"
#include "utl/parser/file.h"

#include "utl/to_vec.h"

namespace motis::parking {

parkings::parkings(std::string const& filename) {
  auto buf = utl::file(filename.c_str(), "r").content();
  utl::cstr s(buf.data(), buf.size());

  using entry = std::tuple<double, double, int>;
  std::vector<entry> entries;
  read(s, entries, {{"lat", "lng", "fee"}});

  auto id = 1;
  parkings_ = utl::to_vec(entries, [&id](entry const& e) {
    return parking_lot(id++, geo::latlng(std::get<0>(e), std::get<1>(e)),
                       std::get<2>(e) != 0);
  });
  rtree_ = geo::make_point_rtree(parkings_,
                                 [](auto const& p) { return p.location_; });
}

parkings::parkings(std::vector<motis::parking::parking_lot>&& parkings)
    : parkings_(std::move(parkings)) {
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
