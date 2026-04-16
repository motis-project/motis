#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

#include "./util.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;

constexpr auto const kSlowDirectGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S1,ICE,,
ICE,S1,ICE2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,00:35:00,00:35:00,DA_10,0,0,1
ICE,00:45:00,00:45:00,FFM_10,1,1,0
ICE2,00:35:00,00:35:00,DA_10,0,0,1
ICE2,00:45:00,00:45:00,FFM_10,1,1,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
ICE,00:35:00,24:35:00,3600
ICE2,00:35:00,24:35:00,3600
)"sv;

TEST(motis, routing_slow_direct) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .tiles_ = {{.profile_ = "deps/tiles/profile/full.lua",
                  .db_size_ = 1024U * 1024U * 25U}},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .use_osm_stop_coordinates_ = true,
              .extend_missing_footpaths_ = false,
              .datasets_ = {{"test", {.path_ = std::string{kSlowDirectGTFS}}}}},
      .gbfs_ = {{.feeds_ = {{"CAB", {.url_ = "./test/resources/gbfs"}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true,
      .geocoding_ = true,
      .reverse_geocoding_ = true};

  import(c, "test/data_osm_only");
  auto d = data{"test/data_osm_only", c};

  auto const routing = utl::init_from<ep::routing>(d).value();

  {
    auto const res = routing(
        "?fromPlace=49.87336,8.62926"
        "&toPlace=test_FFM_10"
        "&time=2019-05-01T01:30Z"
        "&slowDirect=false");
    ASSERT_TRUE(res.itineraries_.size() >= 2);
    EXPECT_EQ(res.itineraries_.at(0).legs_.at(1).tripId_,
              "20190501_03:35_test_ICE2");
    EXPECT_EQ(res.itineraries_.at(1).legs_.at(1).tripId_,
              "20190501_04:35_test_ICE2");
  }
  {
    auto const res = routing(
        "?fromPlace=49.87336,8.62926"
        "&toPlace=test_FFM_10"
        "&time=2019-05-01T01:30Z"
        "&slowDirect=true");
    ASSERT_TRUE(res.itineraries_.size() >= 2);
    EXPECT_EQ(res.itineraries_.at(0).legs_.at(1).tripId_,
              "20190501_03:35_test_ICE2");
    EXPECT_EQ(
        res.itineraries_.at(1).legs_.at(1).tripId_,
        "20190501_03:35_test_ICE2");  // this contains an addional 1min footpath
    EXPECT_EQ(res.itineraries_.at(2).legs_.at(1).tripId_,
              "20190501_03:35_test_ICE");
  }
  {
    auto const res = routing(
        "?fromPlace=49.87336,8.62926"
        "&toPlace=test_FFM_10"
        "&time=2019-05-01T01:30Z"
        "&slowDirect=true"
        "&arriveBy=true"
        "&numItineraries=2&maxItineraries=2");
    ASSERT_TRUE(res.itineraries_.size() >= 2);
    EXPECT_EQ(res.itineraries_.at(0).legs_.at(1).tripId_,
              "20190501_02:35_test_ICE2");
    EXPECT_EQ(res.itineraries_.at(1).legs_.at(1).tripId_,
              "20190501_02:35_test_ICE");
  }
  {
    auto const res = routing(
        "?fromPlace=49.87336,8.62926"
        "&toPlace=test_FFM_10"
        "&time=2019-05-01T01:30Z"
        "&slowDirect=true"
        "&numItineraries=2&maxItineraries=2"
        "&pageCursor=EARLIER%7C1556674200");
    ASSERT_TRUE(res.itineraries_.size() >= 2);
    EXPECT_EQ(res.itineraries_.at(0).legs_.at(1).tripId_,
              "20190501_02:35_test_ICE2");
    EXPECT_EQ(res.itineraries_.at(1).legs_.at(1).tripId_,
              "20190501_02:35_test_ICE");
  }
  {
    auto const res = routing(
        "?fromPlace=49.87336,8.62926"
        "&toPlace=test_FFM_10"
        "&time=2019-05-01T01:30Z"
        "&slowDirect=true"
        "&arriveBy=true"
        "&numItineraries=2&maxItineraries=2"
        "&pageCursor=LATER%7C1556674200");
    ASSERT_TRUE(res.itineraries_.size() >= 2);
    EXPECT_EQ(res.itineraries_.at(0).legs_.at(1).tripId_,
              "20190501_03:35_test_ICE2");
    EXPECT_EQ(res.itineraries_.at(1).legs_.at(1).tripId_,
              "20190501_03:35_test_ICE");
  }
}
