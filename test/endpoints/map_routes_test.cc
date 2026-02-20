#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <chrono>

#include "boost/json.hpp"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/map/routes.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"

namespace json = boost::json;
using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace testing;
namespace n = nigiri;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
DA_Bus_1,DA Hbf,49.8724891,8.6281994
DA_Bus_2,DA Hbf,49.8750407,8.6312172
DA_Tram_1,DA Hbf,49.8742551,8.6321063
DA_Tram_2,DA Hbf,49.8731133,8.6313674
DA_Tram_3,DA Hbf,49.872435,8.632164

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
B1,Test,B1,,3
T1,Test,T1,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign
B1,S1,B1,Bus 1,
T1,S1,T1,Tram 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
B1,01:00:00,01:00:00,DA_Bus_1,1
B1,01:10:00,01:10:00,DA_Bus_2,2
T1,01:05:00,01:05:00,DA_Tram_1,1
T1,01:15:00,01:15:00,DA_Tram_2,2
T1,01:20:00,01:20:00,DA_Tram_3,3

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, map_routes) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{.first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .with_shapes_ = true,
                            .datasets_ = {{"test", {.path_ = kGTFS}}},
                            .route_shapes_ = {{.missing_shapes_ = true,
                                               .replace_shapes_ = true}}},
      .street_routing_ = true};
  auto d = import(c, "test/data", true);

  auto const map_routes = utl::init_from<ep::routes>(d).value();

  {
    auto const res = map_routes(
        "/api/experimental/map/routes"
        "?max=49.88135900212875%2C8.60917200508915"
        "&min=49.863844157325886%2C8.649823169526556"
        "&zoom=16");
    EXPECT_EQ(res.routes_.size(), 2U);
    EXPECT_EQ(res.zoomFiltered_, false);

    EXPECT_THAT(res.routes_, Contains(Field(&api::RouteInfo::mode_,
                                            Eq(api::ModeEnum::BUS))));
    EXPECT_THAT(res.routes_, Contains(Field(&api::RouteInfo::mode_,
                                            Eq(api::ModeEnum::TRAM))));
    EXPECT_THAT(res.routes_, Each(Field(&api::RouteInfo::pathSource_,
                                        Eq(api::RoutePathSourceEnum::ROUTED))));
  }

  {
    // map section without data
    auto const res = map_routes(
        "/api/experimental/map/routes"
        "?max=53.5757876577963%2C9.904453881311966"
        "&min=53.518462458295005%2C10.04877290275494"
        "&zoom=14.5");
    EXPECT_EQ(res.routes_.size(), 0U);
    EXPECT_EQ(res.zoomFiltered_, false);
  }
}
