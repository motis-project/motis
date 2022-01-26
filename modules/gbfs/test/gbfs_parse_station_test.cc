#include "gtest/gtest.h"

#include "motis/gbfs/station.h"

using namespace motis::gbfs;

TEST(gbfs, not_an_array) {
  constexpr auto const in = R"([{
  "station_id":"FE98EABF45DABABA36DEAF660FDA43983E275FEE",
  "lat":48.7829,
  "lon":9.17978,
  "name":"Lautenschlagerstraße / Zeppelin Carré"
  }
])";
  ASSERT_THROW(parse_stations(in), std::runtime_error);
}

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
    "station_id": "FF6424035F35E375A3B54949124B5CAD297CE732",
    "lon": 9.14213,
    "name": "S Bhf. Leinfelden / Bahnhofstr."
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

  auto const stations = parse_stations(in);
  ASSERT_EQ(3, stations.size());

  EXPECT_EQ(stations.at(0),
            (station{.id_ = "FE98EABF45DABABA36DEAF660FDA43983E275FEE",
                     .name_ = "Lautenschlagerstraße / Zeppelin Carré",
                     .pos_ = {48.7829, 9.17978}}));
  EXPECT_EQ(stations.at(1),
            (station{.id_ = "c39d5bf3-e568-43ea-8e37-586878baa368",
                     .name_ = "Uhlbacher Platz/Herrengasse",
                     .pos_ = {48.77605, 9.27931}}));
  EXPECT_EQ(stations.at(2),
            (station{.id_ = "FF6424035F35E375A3B54949124B5CAD297CE732",
                     .name_ = "S Bhf. Leinfelden / Bahnhofstr.",
                     .pos_ = {48.69714, 9.14213}}));
}
