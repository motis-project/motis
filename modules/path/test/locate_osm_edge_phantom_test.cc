#include "gtest/gtest.h"

#include "motis/path/prepare/osm/osm_phantom.h"

namespace mp = motis::path;

TEST(locate_osm_edge_phantom_with_dist, lueneburg) {
  geo::latlng north{53.2510503, 10.4198096};  // north
  geo::latlng south{53.2484046, 10.4197946};  // south
  geo::latlng station_pos{53.249700, 10.419106};  // western / middle

  mp::osm_edge_phantom_match obj{mp::osm_edge_phantom{0, 0, north, south}, 0,
                                 nullptr, station_pos};
  obj.locate();

  ASSERT_FALSE(obj.eq_from_);
  ASSERT_FALSE(obj.eq_to_);

  auto const diff = geo::distance(north, south) -
                    geo::distance(north, obj.pos_) -
                    geo::distance(obj.pos_, south);
  ASSERT_TRUE(std::fabs(diff) < .1);
}

TEST(locate_osm_edge_phantom_with_dist, hamburg) {
  geo::latlng north{53.5530362, 10.0058828};  // north
  geo::latlng south{53.5525599, 10.0061187};  // south
  geo::latlng station_pos{53.552736, 10.006909};  // eastern / northern

  mp::osm_edge_phantom_match obj{mp::osm_edge_phantom{0, 0, south, north}, 0,
                                 nullptr, station_pos};
  std::clog.precision(11);
  obj.locate();

  ASSERT_FALSE(obj.eq_from_);
  ASSERT_FALSE(obj.eq_to_);

  auto const diff = geo::distance(north, south) -
                    geo::distance(north, obj.pos_) -
                    geo::distance(obj.pos_, south);
  ASSERT_TRUE(std::fabs(diff) < .1);
}
