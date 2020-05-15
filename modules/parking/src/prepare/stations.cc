#include "motis/parking/prepare/stations.h"

#include "utl/to_vec.h"

namespace motis::parking::prepare {

stations::stations(schedule const& sched) {
  stations_ = utl::to_vec(sched.stations_, [](auto const& st) {
    return station{st->eva_nr_.str(), geo::latlng{st->lat(), st->lng()}};
  });
  geo_index_ =
      geo::make_point_rtree(stations_, [](auto const& s) { return s.pos_; });
}

std::vector<std::pair<station, double>> stations::get_in_radius(
    geo::latlng const& center, double radius) const {
  return utl::to_vec(
      geo_index_.in_radius_with_distance(center, radius), [&](auto const& s) {
        return std::make_pair(stations_[std::get<1>(s)], std::get<0>(s));
      });
}

}  // namespace motis::parking::prepare
