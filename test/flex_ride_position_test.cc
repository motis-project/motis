#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string_view>
#include <system_error>

#include "fmt/format.h"

#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace motis;
using namespace date;
namespace fs = std::filesystem;

// End to end: the FLEX leg the API prints starts inside the pickup window,
// for rides from the real osr chain (walk to the vehicle first, so the ride
// position and its flooring in add_flex_td_offsets matter).
//
// test_case.osm.pbf has a short drivable stretch west of DA Hbf: from
// Lise-Meitner-Straße (da_south) north to Dornheimer Weg (da_north). From
// Traubenweg (kTraubenweg, outside both areas) it is a ~6 min walk to
// da_south; from DA_10 ~12 min to da_north.
//   FLEX_FM: pickup da_south -> drop-off da_north (first mile to DA_10)
//   FLEX_LM: pickup da_north -> drop-off da_south (last mile from DA_10)
// Both with pickup window {0} and drop-off window {1} (local time).
// ICE_OUT DA_10 11:00 -> FFM_10 12:00, ICE_IN FFM_10 08:40 -> DA_10 09:40
// (Europe/Berlin, UTC+2 in May).
constexpr auto kRidePositionGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101
FLEX,DB,FlexDA,,,715

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S_ALL,ICE_OUT,,
ICE,S_ALL,ICE_IN,,
FLEX,S_ALL,FLEX_FM,,
FLEX,S_ALL,FLEX_LM,,

# booking_rules.txt
booking_rule_id,booking_type,prior_notice_duration_min,prior_notice_duration_max
BR,1,0,86400

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,location_group_id,location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_booking_rule_id,drop_off_booking_rule_id,pickup_type,drop_off_type
ICE_OUT,11:00:00,11:00:00,DA_10,,,0,,,,,0,0
ICE_OUT,12:00:00,12:00:00,FFM_10,,,1,,,,,0,0
ICE_IN,08:40:00,08:40:00,FFM_10,,,0,,,,,0,0
ICE_IN,09:40:00,09:40:00,DA_10,,,1,,,,,0,0
FLEX_FM,,,,,da_south,0,{0},BR,,2,1
FLEX_FM,,,,,da_north,1,{1},,BR,1,2
FLEX_LM,,,,,da_north,0,{0},BR,,2,1
FLEX_LM,,,,,da_south,1,{1},,BR,1,2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S_ALL,1,1,1,1,1,1,1,20190501,20190503

# locations.geojson
{{"type":"FeatureCollection","features":[{{"id":"da_south","type":"Feature","geometry":{{"type":"Polygon","coordinates":[[[8.6270,49.8715],[8.6290,49.8715],[8.6290,49.8745],[8.6270,49.8745],[8.6270,49.8715]]]}},"properties":{{"stop_name":"DA South"}}}},{{"id":"da_north","type":"Feature","geometry":{{"type":"Polygon","coordinates":[[[8.6268,49.8747],[8.6286,49.8747],[8.6286,49.8760],[8.6268,49.8760],[8.6268,49.8747]]]}},"properties":{{"stop_name":"DA North"}}}}]}}
)";

constexpr auto kTraubenweg = "49.87331,8.62300";
constexpr auto kDa10 = "49.87336,8.62926";
constexpr auto kFfm10 = "50.10593,8.66118";

namespace {

struct fixture {
  fixture(std::string_view const sub_dir,
          std::string_view const pickup,
          std::string_view const drop_off) {
    auto ec = std::error_code{};
    auto const path = fs::path{"test/data/flex_ride_position"} / sub_dir;
    fs::remove_all(path, ec);
    cfg_ = config{
        .osm_ = {"test/resources/test_case.osm.pbf"},
        .timetable_ =
            config::timetable{
                .first_day_ = "2019-05-01",
                .num_days_ = 3,
                .datasets_ = {{"test",
                               {.path_ = fmt::format(
                                    fmt::runtime(kRidePositionGtfs), pickup,
                                    drop_off)}}}},
        .street_routing_ = true,
        .osr_footpath_ = true};
    import(cfg_, path);
    d_ = std::make_unique<data>(path, cfg_);
  }

  api::plan_response plan(std::string const& query) const {
    return utl::init_from<ep::routing>(*d_).value()(query);
  }

  config cfg_;
  std::unique_ptr<data> d_;
};

std::chrono::sys_seconds utc(int const day, int const h, int const m) {
  return sys_days{2019_y / May / day} + std::chrono::hours{h} +
         std::chrono::minutes{m};
}

// Time of day (UTC): the search also returns the next days' connections.
std::chrono::minutes utc_minute_of_day(openapi::date_time_t const& t) {
  return std::chrono::duration_cast<std::chrono::minutes>(
      *t - std::chrono::floor<date::days>(*t));
}

constexpr auto hm(int const h, int const m) {
  return std::chrono::hours{h} + std::chrono::minutes{m};
}

bool any_on_day(std::vector<std::pair<api::Leg const*, api::Leg const*>> const&
                    legs,
                int const day) {
  return std::any_of(begin(legs), end(legs), [&](auto const& x) {
    return std::chrono::floor<date::days>(*x.first->startTime_) ==
           sys_days{2019_y / May / day};
  });
}

// All FLEX legs of the given connections, with the leg before each.
std::vector<std::pair<api::Leg const*, api::Leg const*>> flex_legs(
    std::vector<api::Itinerary> const& connections) {
  auto ret = std::vector<std::pair<api::Leg const*, api::Leg const*>>{};
  for (auto const& j : connections) {
    for (auto i = 0U; i != j.legs_.size(); ++i) {
      if (j.legs_[i].mode_ == api::ModeEnum::FLEX) {
        ret.emplace_back(&j.legs_[i], i == 0U ? nullptr : &j.legs_[i - 1U]);
      }
    }
  }
  return ret;
}

std::string first_mile(std::string_view const time,
                       std::string_view const from = kTraubenweg) {
  return fmt::format(
      "/api/v6/plan?fromPlace={}&toPlace={}&time={}"
      "&preTransitModes=FLEX&maxPreTransitTime=3600&postTransitModes=WALK"
      "&directModes=WALK",
      from, kFfm10, time);
}

std::string last_mile(std::string_view const time) {
  return fmt::format(
      "/api/v6/plan?fromPlace={}&toPlace={}&time={}"
      "&preTransitModes=WALK&postTransitModes=FLEX&maxPostTransitTime=3600"
      "&directModes=WALK",
      kFfm10, kTraubenweg, time);
}

std::string direct(std::string_view const time) {
  return fmt::format(
      "/api/v6/plan?fromPlace={}&toPlace={}&time={}"
      "&directModes=FLEX&maxDirectTime=3600",
      kTraubenweg, kDa10, time);
}

}  // namespace

// [T, T] pickup at 10:10 local (08:10Z), drop-off 10:10-28:59.
TEST(motis, flex_ride_position_fixed_departure) {
  auto const f = fixture{"fixed", "10:10:00,10:10:00", "10:10:00,28:59:00"};

  // First mile: walk to da_south, ride exactly at 08:10Z, ICE_OUT at 09:00Z.
  // Several starts along the street: different walks to the vehicle, so the
  // pickup offset lands on different seconds within its minute.
  for (auto const from : {kTraubenweg, "49.87330,8.62340", "49.87329,8.62382",
                          "49.87328,8.62420", "49.87327,8.62460",
                          "49.87326,8.62490"}) {
    SCOPED_TRACE(from);
    auto const res = f.plan(first_mile("2019-05-01T07:30Z", from));
    auto const legs = flex_legs(res.itineraries_);
    ASSERT_FALSE(legs.empty());
    EXPECT_TRUE(any_on_day(legs, 1));
    for (auto const& [flex, before] : legs) {
      EXPECT_EQ(hm(8, 10), utc_minute_of_day(flex->startTime_));
      EXPECT_EQ("da_south", flex->from_.flexId_.value_or("-"));
      EXPECT_EQ("da_north", flex->to_.flexId_.value_or("-"));
      ASSERT_NE(nullptr, before);
      EXPECT_EQ(api::ModeEnum::WALK, before->mode_);
      EXPECT_EQ(hm(8, 10), utc_minute_of_day(before->endTime_));
      EXPECT_LT(*before->startTime_, *before->endTime_);  // walk first
    }
  }

  // Last mile: ICE_IN at DA_10 07:40Z, walk to da_north, ride at 08:10Z.
  {
    auto const res = f.plan(last_mile("2019-05-01T06:30Z"));
    auto const legs = flex_legs(res.itineraries_);
    ASSERT_FALSE(legs.empty());
    EXPECT_TRUE(any_on_day(legs, 1));
    for (auto const& [flex, before] : legs) {
      EXPECT_EQ(hm(8, 10), utc_minute_of_day(flex->startTime_));
      EXPECT_EQ("da_north", flex->from_.flexId_.value_or("-"));
      EXPECT_EQ("da_south", flex->to_.flexId_.value_or("-"));
      ASSERT_NE(nullptr, before);
      EXPECT_EQ(hm(8, 10), utc_minute_of_day(before->endTime_));
    }
  }

  // Direct: moved to the fixed departure; after it, the next day's.
  {
    auto const legs = flex_legs(f.plan(direct("2019-05-01T07:30Z")).direct_);
    ASSERT_EQ(1U, legs.size());
    EXPECT_EQ(utc(1, 8, 10), *legs.front().first->startTime_);
  }
  {
    auto const legs = flex_legs(f.plan(direct("2019-05-01T08:11Z")).direct_);
    ASSERT_EQ(1U, legs.size());
    EXPECT_EQ(utc(2, 8, 10), *legs.front().first->startTime_);
  }
}

// Regular window: pickup 10:00-10:40 local (08:00-08:40Z).
TEST(motis, flex_ride_position_regular_window) {
  auto const f = fixture{"regular", "10:00:00,10:40:00", "10:00:00,13:00:00"};

  // First mile: the ride starts inside the window (the latest departure
  // that still catches ICE_OUT is chosen, so anywhere up to 08:40Z).
  {
    auto const legs =
        flex_legs(f.plan(first_mile("2019-05-01T07:30Z")).itineraries_);
    ASSERT_FALSE(legs.empty());
    EXPECT_TRUE(any_on_day(legs, 1));
    for (auto const& [flex, before] : legs) {
      EXPECT_GE(utc_minute_of_day(flex->startTime_), hm(8, 0));
      EXPECT_LE(utc_minute_of_day(flex->startTime_), hm(8, 40));
    }
  }

  // Last mile: arriving 07:40Z, the ride starts after the walk, in the
  // window.
  {
    auto const legs =
        flex_legs(f.plan(last_mile("2019-05-01T06:30Z")).itineraries_);
    ASSERT_FALSE(legs.empty());
    EXPECT_TRUE(any_on_day(legs, 1));
    for (auto const& [flex, before] : legs) {
      EXPECT_GE(utc_minute_of_day(flex->startTime_), hm(8, 0));
      EXPECT_LE(utc_minute_of_day(flex->startTime_), hm(8, 40));
    }
  }

  // Direct, depart 09:50 local: walk first, the ride starts when the
  // window opens, 10:00 local.
  {
    auto const legs = flex_legs(f.plan(direct("2019-05-01T07:50Z")).direct_);
    ASSERT_EQ(1U, legs.size());
    EXPECT_EQ(utc(1, 8, 0), *legs.front().first->startTime_);
  }

  // Direct, depart 10:41 local: no ride that day, the next day's window.
  {
    auto const legs = flex_legs(f.plan(direct("2019-05-01T08:41Z")).direct_);
    ASSERT_EQ(1U, legs.size());
    EXPECT_EQ(utc(2, 8, 0), *legs.front().first->startTime_);
  }
}
