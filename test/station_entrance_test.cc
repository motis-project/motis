#include "gtest/gtest.h"

#include <filesystem>
#include <string_view>

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "nigiri/station_paths.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace std::string_view_literals;
using namespace motis;

// DA_10 is reached from the street only through the entrance DA_E (GTFS
// pathways, no stop_access=1): the first mile walks to the entrance and
// continues on the pathway.
constexpr auto const kEntranceGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
DA_E,Haupteingang,49.87260,8.63085,2,DA,
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10

# pathways.txt
pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,length,traversal_time,stair_count,max_slope,min_width,signposted_as,reversed_signposted_as
PW1,DA_E,DA_10,1,1,,180,,,,Gleis 10,Ausgang

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,00:35:00,00:35:00,DA_10,0,0,1
ICE,00:45:00,00:45:00,FFM_10,1,1,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
ICE,00:35:00,24:35:00,3600
)"sv;

TEST(motis, station_entrance_first_mile) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_station_entrance", ec);

  auto const c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kEntranceGTFS}}}}},
      .street_routing_ = true};
  import(c, "test/data_station_entrance");
  auto d = data{"test/data_station_entrance", c};

  ASSERT_NE(nullptr, d.station_paths_);
  auto const& sp = *d.station_paths_;
  ASSERT_EQ(1U, sp.entrances_.size());
  EXPECT_EQ(2U, sp.entrance_paths_.size());
  EXPECT_EQ(1U, sp.entrance_only_.size());

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "?fromPlace=49.87200,8.63150"
      "&toPlace=test_FFM_10"
      "&time=2019-05-01T01:30Z"
      "&slowDirect=false");
  ASSERT_FALSE(res.itineraries_.empty());
  auto const& legs = res.itineraries_.front().legs_;
  ASSERT_GE(legs.size(), 3U);

  // street to the entrance, then the pathway to the platform
  EXPECT_EQ(api::ModeEnum::WALK, legs.at(0).mode_);
  EXPECT_EQ("Haupteingang", legs.at(0).to_.name_);
  EXPECT_EQ(api::ModeEnum::WALK, legs.at(1).mode_);
  EXPECT_EQ("Haupteingang", legs.at(1).from_.name_);
  ASSERT_TRUE(legs.at(1).steps_.has_value());
  EXPECT_TRUE(
      utl::any_of(*legs.at(1).steps_, [](api::StepInstruction const& s) {
        return s.streetName_ == "Gleis 10";
      }));
  EXPECT_EQ(legs.at(1).endTime_, legs.at(2).startTime_);
  EXPECT_EQ("20190501_03:35_test_ICE", legs.at(2).tripId_);
}
