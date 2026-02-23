#include "gtest/gtest.h"

#include <algorithm>
#include <filesystem>

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/adr/geocode.h"
#include "motis/import.h"

using namespace motis;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
G1,Group 1,,0.0,0.0,,0,
G2,Group 2,,0.0,0.0,,0,
A,Stop A,,48.1,11.5,,0,
B,Stop B,,48.2,11.6,,0,
C,Stop C,,48.3,11.7,,0,
D,Stop D,,48.4,11.8,,0,

# stop_group_elements.txt
stop_group_id,stop_id
G1,A
G2,B

# calendar_dates.txt
service_id,date,exception_type
S1,20200101,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RB,DB,RB,,,3
RT,DB,RT,,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RB,S1,TB,RB,
RT,S1,TT,RT,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TB,10:00:00,10:00:00,A,1,0,0
TB,10:30:00,10:30:00,C,2,0,0
TT,11:00:00,11:00:00,B,1,0,0
TT,11:30:00,11:30:00,D,2,0,0
)";

TEST(motis, stop_group_geocoding) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2020-01-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .geocoding_ = true};
  import(c, "test/data", true);
  auto d = data{"test/data", c};

  std::cout << "TYPEAHEAD: " << d.t_.get() << "\n";

  auto const geocode = utl::init_from<ep::geocode>(d).value();

  auto const g1 = geocode("/api/v1/geocode?text=Group%201");
  auto const g1_it = std::find_if(
      begin(g1), end(g1), [](auto const& m) { return m.id_ == "test_G1"; });
  ASSERT_NE(end(g1), g1_it);
  ASSERT_TRUE(g1_it->modes_.has_value());
  EXPECT_NE(end(*g1_it->modes_),
            std::find(begin(*g1_it->modes_), end(*g1_it->modes_),
                      api::ModeEnum::BUS));
  ASSERT_TRUE(g1_it->importance_.has_value());
  EXPECT_GT(*g1_it->importance_, 0.0);

  auto const g2 = geocode("/api/v1/geocode?text=Group%202");
  auto const g2_it = std::find_if(
      begin(g2), end(g2), [](auto const& m) { return m.id_ == "test_G2"; });
  ASSERT_NE(end(g2), g2_it);
  ASSERT_TRUE(g2_it->modes_.has_value());
  EXPECT_NE(end(*g2_it->modes_),
            std::find(begin(*g2_it->modes_), end(*g2_it->modes_),
                      api::ModeEnum::TRAM));
  ASSERT_TRUE(g2_it->importance_.has_value());
  EXPECT_GT(*g2_it->importance_, 0.0);
}
