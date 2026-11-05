#include "gtest/gtest.h"

#include <chrono>

#include "net/bad_request_exception.h"

#include "utl/init_from.h"

#include "nigiri/common/parse_time.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/one_to_many_post.h"
#include "motis/import.h"

using namespace motis;
namespace n = nigiri;

namespace {

// Two bus stops, no OSM.
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,55.0,13.0
B,B,55.1,13.0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R,AG,R,,3

# trips.txt
route_id,service_id,trip_id
R,S1,T

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T,10:00:00,10:00:00,A,1
T,10:30:00,10:30:00,B,2
)";

data& timetable_only() {
  static auto const c = config{.timetable_ = config::timetable{
                                   .first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  static auto d = [&]() {
    auto const path = std::filesystem::path{"test/data_timetable_only"};
    auto ec = std::error_code{};
    std::filesystem::remove_all(path, ec);
    import(c, path);
    return data{path, c};
  }();
  return d;
}

}  // namespace

// Without OSM there is no street routing to start and end with.
TEST(motis, one_to_many_intermodal_without_osm) {
  auto& d = timetable_only();
  auto const ep = utl::init_from<ep::one_to_many_intermodal_post>(d);
  ASSERT_TRUE(ep.has_value());
  EXPECT_THROW((*ep)(api::OneToManyIntermodalParams{
                   .one_ = "test_A",
                   .many_ = {"test_B"},
                   .time_ = std::chrono::time_point_cast<std::chrono::seconds>(
                       n::parse_time("2019-05-01T10:00:00+02:00", "%FT%T%Ez")),
                   .maxTravelTime_ = 120}),
               net::bad_request_exception);
}

// The modes filter the stops, grouped or not.
TEST(motis, map_stops_filter_modes) {
  auto& d = timetable_only();
  auto const stops = utl::init_from<ep::stops>(d).value();
  constexpr auto const kBox =
      "/api/v1/map/stops?min=54.9%2C12.9&max=55.2%2C13.1";
  EXPECT_EQ(2U, stops(kBox).size());
  EXPECT_EQ(2U, stops(fmt::format("{}&modes=BUS", kBox)).size());
  EXPECT_TRUE(stops(fmt::format("{}&modes=SUBWAY", kBox)).empty());
}
