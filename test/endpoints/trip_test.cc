#include "gtest/gtest.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"

#include "utl/init_from.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
Parent1,Parent1,50.0,8.0,1,,
Child1A,Child1A,50.001,8.001,0,Parent1,1
Child1B,Child1B,50.002,8.002,0,Parent1,2
Parent2,Parent2,51.0,9.0,1,,
Child2,Child2,51.001,9.001,0,Parent2,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,Child1A,1,0,0
T1,10:10:00,10:10:00,Child1B,2,0,0
T1,11:00:00,11:00:00,Child2,3,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

TEST(motis, trip_stop_naming) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.timetable_ =
                 config::timetable{
                     .first_day_ = "2019-05-01",
                     .num_days_ = 2,
                     .datasets_ = {{"test", {.path_ = std::string{kGTFS}}}}},
             .street_routing_ = false};
  auto d = import(c, "test/data", true);

  auto const trip_ep = utl::init_from<ep::trip>(d).value();

  auto const res = trip_ep("?tripId=20190501_10%3A00_test_T1");

  ASSERT_EQ(1, res.legs_.size());
  auto const& leg = res.legs_[0];

  // Check start stop name
  EXPECT_EQ("Child1A", leg.from_.name_);

  // Check intermediate stops
  ASSERT_TRUE(leg.intermediateStops_.has_value());
  ASSERT_EQ(1, leg.intermediateStops_->size());
  EXPECT_EQ("Child1B", leg.intermediateStops_->at(0).name_);

  // Check end stop name
  EXPECT_EQ("Parent2", leg.to_.name_);
}
