#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "date/tz.h"

#include "fmt/format.h"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace std::chrono_literals;
using namespace date;
using namespace motis;

namespace {

// Daily flex trip in an area around DA Hbf (08:30-18:00 local time) and a
// daily ICE DA -> FFM -> DA. nigiri stores one flex transport per UTC offset,
// so the flex trip has a CET and a CEST variant.
constexpr auto const kFlexDstGtfs = R"(
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
FLEX,DB,Flex,,,715

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S,ICE_DA_FFM,,
ICE,S,ICE_FFM_DA,,
FLEX,S,FLEX_DA,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type
ICE_DA_FFM,12:00:00,12:00:00,DA_10,,0,,,0,0
ICE_DA_FFM,12:30:00,12:30:00,FFM_10,,1,,,0,0
ICE_FFM_DA,13:00:00,13:00:00,FFM_10,,0,,,0,0
ICE_FFM_DA,13:30:00,13:30:00,DA_10,,1,,,0,0
FLEX_DA,,,,da_flex,0,08:30:00,18:00:00,2,1
FLEX_DA,,,,da_flex,1,08:30:00,18:00:00,1,2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S,1,1,1,1,1,1,1,20190101,20201231

# locations.geojson
{"type":"FeatureCollection","features":[{"id":"da_flex","type":"Feature","geometry":{"type":"Polygon","coordinates":[[[8.620,49.865],[8.640,49.865],[8.640,49.880],[8.620,49.880],[8.620,49.865]]]},"properties":{"stop_name":"DA Flex Area"}}]}
)";

// A day and an evening flex trip with the same stop sequence.
constexpr auto const kFlexTwoTripsGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
DA_10,DA Hbf,49.87336,8.62926

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
FLEX,DB,Flex,,,715

# trips.txt
route_id,service_id,trip_id
FLEX,S,FLEX_DAY
FLEX,S,FLEX_EVENING

# stop_times.txt
trip_id,location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type
FLEX_DAY,da_flex,0,08:30:00,18:00:00,2,1
FLEX_DAY,da_flex,1,08:30:00,18:00:00,1,2
FLEX_EVENING,da_flex,0,19:00:00,23:00:00,2,1
FLEX_EVENING,da_flex,1,19:00:00,23:00:00,1,2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S,1,1,1,1,1,1,1,20190501,20190503

# locations.geojson
{"type":"FeatureCollection","features":[{"id":"da_flex","type":"Feature","geometry":{"type":"Polygon","coordinates":[[[8.620,49.865],[8.640,49.865],[8.640,49.880],[8.620,49.880],[8.620,49.865]]]},"properties":{"stop_name":"DA Flex Area"}}]}
)";

constexpr auto const kInArea = "49.87000,8.62500";
constexpr auto const kDaHbf = "49.87336,8.62926";
constexpr auto const kFfmHbf = "50.10593,8.66118";

sys_seconds local(sys_days const day, std::chrono::minutes const time) {
  return locate_zone("Europe/Berlin")
      ->to_sys(local_days{day.time_since_epoch()} + time);
}

std::string to_str(sys_seconds const t) { return date::format("%FT%H:%MZ", t); }

config flex_config(std::string_view const gtfs,
                   sys_days const first_day,
                   std::uint16_t const num_days) {
  return config{.osm_ = {"test/resources/test_case.osm.pbf"},
                .timetable_ =
                    config::timetable{
                        .first_day_ = date::format("%F", first_day),
                        .num_days_ = num_days,
                        .datasets_ = {{"test", {.path_ = std::string{gtfs}}}}},
                .street_routing_ = true,
                .osr_footpath_ = true};
}

// Every FLEX leg has the windows `from_time` - `to_time` (local time) of its
// day and lies inside them. At least one of them is on `day`.
void expect_windows(std::vector<api::Itinerary> const& itineraries,
                    sys_days const day,
                    std::chrono::minutes const from_time = 8h + 30min,
                    std::chrono::minutes const to_time = 18h) {
  auto on_day = false;
  for (auto const& it : itineraries) {
    for (auto const& leg : it.legs_) {
      if (leg.mode_ != api::ModeEnum::FLEX) {
        continue;
      }
      auto const leg_day = sys_days{
          floor<days>(
              locate_zone("Europe/Berlin")->to_local(leg.startTime_.time_))
              .time_since_epoch()};
      on_day |= leg_day == day;

      auto const from = local(leg_day, from_time);
      auto const to = local(leg_day, to_time);
      for (auto const* p : {&leg.from_, &leg.to_}) {
        ASSERT_TRUE(p->flexStartPickupDropOffWindow_.has_value());
        ASSERT_TRUE(p->flexEndPickupDropOffWindow_.has_value());
        EXPECT_EQ(from, p->flexStartPickupDropOffWindow_->time_);
        EXPECT_EQ(to, p->flexEndPickupDropOffWindow_->time_);
      }
      EXPECT_LE(from, leg.startTime_.time_);
      EXPECT_LE(leg.endTime_.time_, to);
    }
  }
  EXPECT_TRUE(on_day);
}

constexpr auto const kDirect = "transitModes=&directModes=FLEX";
constexpr auto const kFirstMile = "directModes=&preTransitModes=FLEX";
constexpr auto const kLastMile = "directModes=&postTransitModes=FLEX";

struct query {
  char const* label_;
  char const* from_;
  char const* to_;
  std::chrono::minutes time_;  // local time
  bool arrive_by_;
  std::string_view modes_;

  bool is_direct() const { return modes_ == kDirect; }
};

// Direct and as first / last mile, depart at and arrive by.
constexpr auto const kQueries = std::array{
    query{"direct", kInArea, kDaHbf, 10h, false, kDirect},
    query{"direct, arrive by", kInArea, kDaHbf, 10h, true, kDirect},
    query{"first mile", kInArea, kFfmHbf, 11h, false, kFirstMile},
    query{"first mile, arrive by", kInArea, kFfmHbf, 13h, true, kFirstMile},
    query{"last mile", kFfmHbf, kInArea, 12h + 30min, false, kLastMile},
    query{"last mile, arrive by", kFfmHbf, kInArea, 15h, true, kLastMile}};

std::vector<api::Itinerary> route(ep::routing const& routing,
                                  query const& q,
                                  sys_days const day,
                                  bool const timetable_view) {
  auto res = routing(fmt::format(
      "?fromPlace={}&toPlace={}&time={}&arriveBy={}&timetableView={}&{}",
      q.from_, q.to_, to_str(local(day, q.time_)), q.arrive_by_, timetable_view,
      q.modes_));
  return q.is_direct() ? std::move(res.direct_) : std::move(res.itineraries_);
}

// Queries every day from 3 days before to 3 days after each DST change.
// nigiri numbers the variants in the order of their first service day: a
// timetable starting in winter has the CET variant first, one starting in
// summer the CEST variant.
void run_flex_dst_test(std::string_view const name,
                       sys_days const first_day,
                       std::uint16_t const num_days,
                       std::initializer_list<sys_days> const changes) {
  auto const path = fmt::format("test/data_flex_dst_{}", name);
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);

  auto const c = flex_config(kFlexDstGtfs, first_day, num_days);
  import(c, path);
  auto d = data{path, c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  for (auto const change : changes) {
    for (auto day = change - days{3}; day <= change + days{3}; day += days{1}) {
      SCOPED_TRACE(date::format("%F", day));
      for (auto const& q : kQueries) {
        for (auto const timetable_view : {false, true}) {
          SCOPED_TRACE(
              fmt::format("{}, timetableView={}", q.label_, timetable_view));
          expect_windows(route(routing, q, day, timetable_view), day);
        }
      }
    }
  }
}

}  // namespace

TEST(motis, flex_dst_winter_first) {
  run_flex_dst_test(
      "winter_first", sys_days{2019_y / March / 20}, 230U,
      {sys_days{2019_y / March / 31}, sys_days{2019_y / October / 27}});
}

TEST(motis, flex_dst_summer_first) {
  run_flex_dst_test(
      "summer_first", sys_days{2019_y / October / 15}, 180U,
      {sys_days{2019_y / October / 27}, sys_days{2020_y / March / 29}});
}

// Direct FLEX legs get the windows of the trip that runs at the time of the
// leg, not the ones of the first trip with the same stop sequence.
TEST(motis, flex_direct_two_trips) {
  auto const path = "test/data_flex_two_trips";
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);

  auto const day = sys_days{2019_y / May / 2};
  auto const c = flex_config(kFlexTwoTripsGtfs, day - days{1}, 3U);
  import(c, path);
  auto d = data{path, c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  for (auto const arrive_by : {false, true}) {
    SCOPED_TRACE(fmt::format("arriveBy={}", arrive_by));
    auto const at = [&](std::chrono::minutes const time) {
      return route(routing,
                   query{"", kInArea, kDaHbf, time, arrive_by, kDirect}, day,
                   false);
    };
    expect_windows(at(10h), day);
    expect_windows(at(20h), day, 19h, 23h);
  }
}
