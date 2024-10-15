#include "gtest/gtest.h"

#include "motis/gbfs/station.h"

using namespace motis::gbfs;

TEST(gbfs, parse_stations) {
  constexpr auto const in = R"({
"last_updated": 1643129104,
"ttl": 0,
"version": "2.2",
"data": {
"stations": [
  {
    "station_id": "FE98EABF45DABABA36DEAF660FDA43983E275FEE",
    "lat": 48.7829,
    "lon": 9.17978,
    "name": "Lautenschlagerstraße / Zeppelin Carré"
  },
  {
    "station_id": "c39d5bf3-e568-43ea-8e37-586878baa368",
    "lat": 48.77605,
    "lon": 9.27931,
    "name": "Uhlbacher Platz/Herrengasse"
  },
  {
    "station_id": "FF6424035F35E375A3B54949124B5CAD297CE732",
    "lat": 48.69714,
    "lon": 9.14213,
    "name": "S Bhf. Leinfelden / Bahnhofstr."
  }
]
}
})";

  constexpr auto const in1 = R"({
  "last_updated": 1643234179,
  "ttl": 0,
  "version": "2.2",
  "data": {
    "stations": [
      {
        "station_id": "FE98EABF45DABABA36DEAF660FDA43983E275FEE",
        "is_installed": true,
        "last_reported": 1643233951,
        "num_bikes_available": 11,
        "is_renting": true,
        "is_returning": true,
        "vehicle_types_available": [
          {
            "count": 11,
            "vehicle_type_id": "bike"
          }
        ]
      },
      {
        "station_id": "c39d5bf3-e568-43ea-8e37-586878baa368",
        "is_installed": true,
        "last_reported": 1643233951,
        "num_bikes_available": 7,
        "is_renting": true,
        "is_returning": true,
        "vehicle_types_available": [
          {
            "count": 7,
            "vehicle_type_id": "bike"
          }
        ]
      },
      {
        "station_id": "FF6424035F35E375A3B54949124B5CAD297CE732",
        "is_installed": true,
        "last_reported": 1643233951,
        "num_bikes_available": 10,
        "is_renting": true,
        "is_returning": true,
        "vehicle_types_available": [
          {
            "count": 10,
            "vehicle_type_id": "bike"
          }
        ]
      }
    ]
  }
}
)";

  auto const stations = parse_stations("test-", in, in1);
  ASSERT_EQ(3, stations.size());

  EXPECT_EQ(stations.at("test-FE98EABF45DABABA36DEAF660FDA43983E275FEE"),
            (station{.id_ = "test-FE98EABF45DABABA36DEAF660FDA43983E275FEE",
                     .name_ = "Lautenschlagerstraße / Zeppelin Carré",
                     .pos_ = {48.7829, 9.17978},
                     .bikes_available_ = 11}));
  EXPECT_EQ(stations.at("test-c39d5bf3-e568-43ea-8e37-586878baa368"),
            (station{.id_ = "test-c39d5bf3-e568-43ea-8e37-586878baa368",
                     .name_ = "Uhlbacher Platz/Herrengasse",
                     .pos_ = {48.77605, 9.27931},
                     .bikes_available_ = 7}));
  EXPECT_EQ(stations.at("test-FF6424035F35E375A3B54949124B5CAD297CE732"),
            (station{.id_ = "test-FF6424035F35E375A3B54949124B5CAD297CE732",
                     .name_ = "S Bhf. Leinfelden / Bahnhofstr.",
                     .pos_ = {48.69714, 9.14213},
                     .bikes_available_ = 10}));
}
