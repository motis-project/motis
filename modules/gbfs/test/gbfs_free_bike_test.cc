#include "gtest/gtest.h"

#include "motis/gbfs/free_bike.h"

using namespace motis::gbfs;

TEST(gbfs, parse_free_bikes) {
  constexpr auto const in = R"({
  "last_updated": 1643129083,
  "ttl": 0,
  "version": "2.2",
  "data": {
    "bikes": [
      {
        "lat": 48.7829,
        "lon": 9.17978,
        "bike_id": "184183",
        "is_reserved": false,
        "is_disabled": false,
        "vehicle_type_id": "bike",
        "last_reported": 1643128951
      },
      {
        "lat": 48.7829,
        "lon": 9.17978,
        "bike_id": "184369",
        "is_reserved": false,
        "is_disabled": false,
        "vehicle_type_id": "bike",
        "last_reported": 1643128951
      }
    ]
  }
})";

  auto const bikes = parse_free_bikes("yeah-", in);
  ASSERT_EQ(2, bikes.size());

  EXPECT_EQ(bikes.at(0), (free_bike{.id_ = "yeah-184183",
                                    .pos_ = {48.7829, 9.17978},
                                    .type_ = "bike"}));
  EXPECT_EQ(bikes.at(1), (free_bike{.id_ = "yeah-184369",
                                    .pos_ = {48.7829, 9.17978},
                                    .type_ = "bike"}));
}
