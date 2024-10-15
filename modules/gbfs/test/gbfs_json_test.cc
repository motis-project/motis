#include "gtest/gtest.h"

#include "motis/gbfs/system_status.h"

using namespace motis::gbfs;

constexpr auto const in = R"({
  "last_updated": 1640887163,
  "ttl": 0,
  "version": "3.0",
  "data": {
    "en": {
      "feeds": [
        {
          "name": "system_information",
          "url": "https://www.example.com/gbfs/1/en/system_information"
        },
        {
          "name": "station_information",
          "url": "https://www.example.com/gbfs/1/en/station_information"
        }
      ]
    },
    "fr" : {
      "feeds": [
        {
          "name": "system_information",
          "url": "https://www.example.com/gbfs/1/fr/system_information"
        },
        {
          "name": "station_information",
          "url": "https://www.example.com/gbfs/1/fr/station_information"
        }
      ]
    },
    "de":{
      "feeds":[
        {
          "name":"free_bike_status",
          "url":"https://127.0.0.1/gbfs/v2/free_bike_status.json"
        },
        {
          "name":"vehicle_types",
          "url":"https://127.0.0.1/gbfs/v2/vehicle_types.json"
        },
        {
          "name":"station_information",
          "url":"https://127.0.0.1/gbfs/v2/station_information.json"
        },
        {
          "name":"system_information",
          "url":"https://127.0.0.1/gbfs/v2/system_information.json"
        },
        {
          "name":"station_status",
          "url":"https://127.0.0.1/gbfs/v2/station_status.json"
        }
      ]
    }
  }
})";

TEST(gbfs, parse_gbfs_json) {
  auto const urls = read_system_status(in);
  ASSERT_EQ(3, urls.size());

  ASSERT_TRUE(urls.at(0).station_info_url_);
  EXPECT_EQ("https://www.example.com/gbfs/1/en/station_information",
            urls.at(0).station_info_url_);

  ASSERT_TRUE(urls.at(1).station_info_url_);
  EXPECT_EQ("https://www.example.com/gbfs/1/fr/station_information",
            urls.at(1).station_info_url_);

  ASSERT_TRUE(urls.at(2).free_bike_url_);
  EXPECT_EQ("https://127.0.0.1/gbfs/v2/free_bike_status.json",
            urls.at(2).free_bike_url_);

  ASSERT_TRUE(urls.at(2).station_info_url_);
  EXPECT_EQ("https://127.0.0.1/gbfs/v2/station_information.json",
            urls.at(2).station_info_url_);

  ASSERT_TRUE(urls.at(2).station_status_url_);
  EXPECT_EQ("https://127.0.0.1/gbfs/v2/station_status.json",
            urls.at(2).station_status_url_);
}
