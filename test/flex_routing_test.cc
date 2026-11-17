#include "gtest/gtest.h"

#include <filesystem>

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace std::string_view_literals;
using namespace motis;

// E is on a separately mapped sidewalk, 6 m from the road the vehicle uses.
constexpr auto const kFlexSidewalkGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
FA,Flex Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Flex A,50.1062497,8.6582223
E,Flex E,50.1079161,8.6605383

# location_groups.txt
location_group_id,location_group_name
LG,Flex Zone

# location_group_stops.txt
location_group_id,stop_id
LG,A
LG,E

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FR,FA,Rufbus,,715

# trips.txt
route_id,service_id,trip_id
FR,S1,FT

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,location_group_id,location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type
FT,,,,LG,,1,06:00:00,22:00:00,2,2
FT,,,,LG,,2,06:00:00,22:00:00,2,2
)"sv;

TEST(motis, flex_drop_off_at_sidewalk_stop) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_flex_sidewalk", ec);

  auto const c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"flex",
                             {.path_ = std::string{kFlexSidewalkGtfs}}}}},
      .street_routing_ = true};
  import(c, "test/data_flex_sidewalk");
  auto d = data{"test/data_flex_sidewalk", c};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "?fromPlace=50.1062497,8.6582223"
      "&toPlace=50.1079161,8.6605383"
      "&time=2019-05-01T08:00Z"
      "&timetableView=false"
      "&transitModes="
      "&directModes=FLEX");

  ASSERT_EQ(1U, res.direct_.size());
  auto const& legs = res.direct_.front().legs_;
  auto const flex = utl::find_if(
      legs, [](auto const& l) { return l.mode_ == api::ModeEnum::FLEX; });
  ASSERT_NE(end(legs), flex);
  EXPECT_EQ("Flex A", flex->from_.name_);
  EXPECT_EQ("Flex E", flex->to_.name_);
}
