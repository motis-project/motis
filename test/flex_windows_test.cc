#include <chrono>
#include <filesystem>
#include <string_view>
#include <system_error>

#include "fmt/format.h"

#include "gtest/gtest.h"

#include "nigiri/timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/flex/flex.h"
#include "motis/import.h"
#include "motis/point_rtree.h"

using namespace motis;
using namespace std::chrono_literals;
using namespace date;
namespace fs = std::filesystem;
namespace n = nigiri;

namespace {

// ICE FFM_10 -> DA_10 arrives 10:00 local (08:00Z). Flex service FLEX_DA:
// pickup only in area da_pickup (around DA Hbf) in the first {}, drop-off
// only in area da_area (west of it) in the second {} (default 10:00-13:00
// local, 08:00-11:00Z).
constexpr auto kFlexWindowsGtfs = R"(
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
ICE,S_ALL,ICE,,
FLEX,S_ALL,FLEX_DA,,

# booking_rules.txt
booking_rule_id,booking_type,prior_notice_duration_min,prior_notice_duration_max
BR,1,0,86400

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,location_group_id,location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_booking_rule_id,drop_off_booking_rule_id,pickup_type,drop_off_type
ICE,09:00:00,09:00:00,FFM_10,,,0,,,,,0,0
ICE,10:00:00,10:00:00,DA_10,,,1,,,,,0,0
FLEX_DA,,,,,da_pickup,0,{},BR,,2,1
FLEX_DA,,,,,da_area,1,{},,BR,1,2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S_ALL,1,1,1,1,1,1,1,20190501,20190503

# locations.geojson
{{"type":"FeatureCollection","features":[{{"id":"da_pickup","type":"Feature","geometry":{{"type":"Polygon","coordinates":[[[8.625,49.870],[8.635,49.870],[8.635,49.876],[8.625,49.876],[8.625,49.870]]]}},"properties":{{"stop_name":"DA Pickup"}}}},{{"id":"da_area","type":"Feature","geometry":{{"type":"Polygon","coordinates":[[[8.610,49.865],[8.624,49.865],[8.624,49.880],[8.610,49.880],[8.610,49.865]]]}},"properties":{{"stop_name":"DA Flex Area"}}}}]}}

)";

constexpr auto kPickup = geo::latlng{49.87336, 8.62926};  // DA_10, da_pickup
constexpr auto kDest = geo::latlng{49.8755, 8.6185};  // da_area

data load(std::string_view const sub_dir,
          std::string_view const pickup,
          std::string_view const drop_off = "10:00:00,13:00:00") {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/flex_windows"} / sub_dir;
  fs::remove_all(path, ec);
  auto const cfg = config{
      .timetable_ = config::timetable{
          .first_day_ = "2019-05-01",
          .num_days_ = 3,
          .datasets_ = {{"test",
                         {.path_ = fmt::format(fmt::runtime(kFlexWindowsGtfs),
                                               pickup, drop_off)}}}}};
  import(cfg, path);
  return data{path, cfg};
}

n::unixtime_t utc(int const day, int const h, int const m) {
  return n::unixtime_t{sys_days{2019_y / May / day}} + std::chrono::hours{h} +
         std::chrono::minutes{m};
}

flex::flex_routings_t routings(data const& d,
                               geo::latlng const& pos,
                               osr::direction const dir,
                               std::chrono::seconds const max = 15min) {
  return flex::get_flex_routings(*d.tt_, *d.location_rtree_, utc(1, 7, 30),
                                 pos, dir, max, osr_parameters{});
}

// The only flex transport; operating on 2019-05-01 (UTC traffic day).
flex::mode_payload offer() {
  return flex::mode_payload{n::flex_transport_idx_t{0U}, 0U, 1U};
}

flex::flex_ride ride(n::duration_t const pickup, n::duration_t const drop_off) {
  return flex::flex_ride{.pickup_ = pickup, .drop_off_ = drop_off};
}

api::Itinerary itinerary(n::unixtime_t const start,
                         n::unixtime_t const end,
                         api::ModeEnum const mode) {
  auto leg = api::Leg{};
  leg.mode_ = mode;
  leg.startTime_ = start;
  leg.endTime_ = end;
  leg.scheduledStartTime_ = start;
  leg.scheduledEndTime_ = end;
  leg.from_.departure_ = start;
  leg.to_.arrival_ = end;
  auto j = api::Itinerary{};
  j.startTime_ = start;
  j.endTime_ = end;
  j.legs_ = {leg};
  return j;
}

}  // namespace

// The small timetable fits the payload; a transport with more stop rows than
// the payload can address is rejected at import.
TEST(motis, flex_limits) {
  auto const d = load("limits", "10:10:00,10:40:00");
  EXPECT_NO_THROW(flex::verify_flex_limits(*d.tt_));

  auto rows = std::string{};
  for (auto i = 0U; i != (1U << flex::mode_payload::kStopBits) + 1U; ++i) {
    rows += fmt::format("FLEX_DA,,,,,{},{},10:00:00,13:00:00,BR,BR,2,2\n",
                        i % 2 == 0 ? "da_pickup" : "da_area", i);
  }
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/flex_windows/limits_exceeded"};
  fs::remove_all(path, ec);
  auto const gtfs = fmt::format(fmt::runtime(kFlexWindowsGtfs),
                                "10:00:00,13:00:00", "10:00:00,13:00:00");
  auto const begin = gtfs.find("FLEX_DA,,,,,da_pickup");
  auto const end = gtfs.find("\n# calendar.txt");
  auto const cfg = config{
      .timetable_ = config::timetable{
          .first_day_ = "2019-05-01",
          .num_days_ = 3,
          .datasets_ = {{"test",
                         {.path_ = gtfs.substr(0, begin) + rows +
                                   gtfs.substr(end)}}}}};
  EXPECT_THROW(import(cfg, path), std::runtime_error);
}

// The ride always runs from boarding (index 0, da_pickup) to alighting stop
// (index 1, da_area): the first mile boards at the query position, the last
// mile alights there. Never the other way round.
TEST(motis, flex_routings_travel_order) {
  auto const d = load("routings", "10:10:00,10:40:00");

  auto const key = std::pair{n::flex_stop_seq_idx_t{0U},
                             std::pair{n::stop_idx_t{0U}, n::stop_idx_t{1U}}};

  auto const first_mile = routings(d, kPickup, osr::direction::kForward);
  ASSERT_EQ(1U, first_mile.size());
  ASSERT_TRUE(first_mile.contains(key));
  auto const& fm = first_mile.at(key);
  ASSERT_EQ(1U, fm.size());
  EXPECT_EQ(0U, fm.front().get_from_stop());
  EXPECT_EQ(1U, fm.front().get_to_stop());

  auto const last_mile = routings(d, kDest, osr::direction::kBackward);
  ASSERT_EQ(1U, last_mile.size());
  ASSERT_TRUE(last_mile.contains(key));
  auto const& lm = last_mile.at(key);
  ASSERT_EQ(1U, lm.size());
  EXPECT_EQ(0U, lm.front().get_from_stop());
  EXPECT_EQ(1U, lm.front().get_to_stop());

  // Drop-off only area: no first mile from there, pickup only area: no last
  // mile to there (with a walking budget too short to reach the other area).
  EXPECT_TRUE(routings(d, kDest, osr::direction::kForward, 1min).empty());
  EXPECT_TRUE(routings(d, kPickup, osr::direction::kBackward, 1min).empty());
}

// W = [a_from - pickup, b_from - pickup] ∩ [a_to - drop_off, b_to - drop_off],
// both ends inclusive, returned as [start, end + 1 min).
// Pickup 08:10-08:40Z, drop-off 08:00-11:00Z.
TEST(motis, flex_departure_window) {
  auto const d = load("window", "10:10:00,10:40:00");
  auto const day = utc(1, 0, 0);
  auto const id = offer();

  EXPECT_EQ((n::interval{utc(1, 8, 10), utc(1, 8, 41)}),
            flex::get_departure_window(*d.tt_, id, day, ride(0min, 20min)));
  // Walking 5 min to the vehicle: leave 5 min before the pickup window.
  EXPECT_EQ((n::interval{utc(1, 8, 5), utc(1, 8, 36)}),
            flex::get_departure_window(*d.tt_, id, day, ride(5min, 25min)));
  // The drop-off window closes first. A walk after the drop-off does not
  // count.
  EXPECT_EQ((n::interval{utc(1, 8, 10), utc(1, 8, 31)}),
            flex::get_departure_window(*d.tt_, id, day, ride(0min, 150min)));
  auto const none =
      flex::get_departure_window(*d.tt_, id, day, ride(0min, 180min));
  EXPECT_GE(none.from_, none.to_);
}

// Zero-length pickup window [T, T]: a fixed departure at minute T (call-taxi
// encoding of the Austrian feeds), here with a drop-off window from the same
// minute to the end of the service day (28:59).
TEST(motis, flex_zero_length_window_is_fixed_departure) {
  auto const d = load("zero_window", "10:10:00,10:10:00", "10:10:00,28:59:00");
  auto const id = offer();
  auto const day = utc(1, 0, 0);

  // Exactly one departure minute: pickup at 08:10Z.
  EXPECT_EQ((n::interval{utc(1, 8, 10), utc(1, 8, 11)}),
            flex::get_departure_window(*d.tt_, id, day, ride(0min, 10min)));
  // 3 min walk to the vehicle: leave at 08:07Z.
  EXPECT_EQ((n::interval{utc(1, 8, 7), utc(1, 8, 8)}),
            flex::get_departure_window(*d.tt_, id, day, ride(3min, 13min)));
  // Drop-off until 28:59 local = 02:59Z the next day, inclusive.
  EXPECT_EQ((n::interval{utc(1, 8, 10), utc(1, 8, 11)}),
            flex::get_departure_window(*d.tt_, id, day,
                                       ride(0min, 1129min)));  // 02:59Z
  auto const late =
      flex::get_departure_window(*d.tt_, id, day, ride(0min, 1130min));
  EXPECT_GE(late.from_, late.to_);

  auto const ids = std::vector{id};
  // Direct, depart 07:30: moved to the fixed departure.
  {
    auto j = itinerary(utc(1, 7, 30), utc(1, 7, 45), api::ModeEnum::FLEX);
    ASSERT_TRUE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 7, 30),
                                            false, j));
    EXPECT_EQ(utc(1, 8, 10), *j.startTime_);
    EXPECT_EQ(utc(1, 8, 25), *j.endTime_);
    auto const& l = j.legs_.front();
    EXPECT_EQ(utc(1, 8, 10), **l.from_.flexStartPickupDropOffWindow_);
    EXPECT_EQ(utc(1, 8, 10), **l.from_.flexEndPickupDropOffWindow_);
    EXPECT_EQ(utc(2, 2, 59), **l.to_.flexEndPickupDropOffWindow_);
  }
  // Direct, arrive by 12:00: still the 08:10 departure.
  {
    auto j = itinerary(utc(1, 11, 45), utc(1, 12, 0), api::ModeEnum::FLEX);
    ASSERT_TRUE(
        flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 12, 0), true, j));
    EXPECT_EQ(utc(1, 8, 10), *j.startTime_);
  }
  // Depart at 08:11: the next departure is the next day's.
  {
    auto j = itinerary(utc(1, 8, 11), utc(1, 8, 26), api::ModeEnum::FLEX);
    ASSERT_TRUE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 8, 11),
                                            false, j));
    EXPECT_EQ(utc(2, 8, 10), *j.startTime_);
  }
}

// Pickup 08:10-08:40Z, drop-off 08:00-11:00Z, operating 2019-05-01..03.
TEST(motis, flex_direct_respects_windows) {
  auto const d = load("direct", "10:10:00,10:40:00");
  auto const ids = std::vector{offer()};

  // Depart at 07:30: moved to the first departure, 08:10.
  {
    auto j = itinerary(utc(1, 7, 30), utc(1, 7, 45), api::ModeEnum::FLEX);
    ASSERT_TRUE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 7, 30),
                                            false, j));
    EXPECT_EQ(utc(1, 8, 10), *j.startTime_);
    EXPECT_EQ(utc(1, 8, 25), *j.endTime_);
    auto const& l = j.legs_.front();
    EXPECT_EQ(utc(1, 8, 10), *l.startTime_);
    EXPECT_EQ(utc(1, 8, 25), *l.endTime_);
    EXPECT_EQ(utc(1, 8, 10), **l.from_.departure_);
    EXPECT_EQ(utc(1, 8, 25), **l.to_.arrival_);
    EXPECT_EQ(utc(1, 8, 10), **l.from_.flexStartPickupDropOffWindow_);
    EXPECT_EQ(utc(1, 8, 40), **l.from_.flexEndPickupDropOffWindow_);
    EXPECT_EQ(utc(1, 8, 0), **l.to_.flexStartPickupDropOffWindow_);
    EXPECT_EQ(utc(1, 11, 0), **l.to_.flexEndPickupDropOffWindow_);
  }

  // Depart at 08:20, inside the window: unchanged.
  {
    auto j = itinerary(utc(1, 8, 20), utc(1, 8, 35), api::ModeEnum::FLEX);
    ASSERT_TRUE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 8, 20),
                                            false, j));
    EXPECT_EQ(utc(1, 8, 20), *j.startTime_);
  }

  // Depart at 08:41: the window has closed, the next day's departure.
  {
    auto j = itinerary(utc(1, 8, 41), utc(1, 8, 56), api::ModeEnum::FLEX);
    ASSERT_TRUE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 8, 41),
                                            false, j));
    EXPECT_EQ(utc(2, 8, 10), *j.startTime_);
    EXPECT_EQ(utc(2, 8, 10),
              **j.legs_.front().from_.flexStartPickupDropOffWindow_);
  }

  // Arrive by 12:00: latest departure 08:40 (window end is inclusive).
  {
    auto j = itinerary(utc(1, 11, 45), utc(1, 12, 0), api::ModeEnum::FLEX);
    ASSERT_TRUE(
        flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 12, 0), true, j));
    EXPECT_EQ(utc(1, 8, 40), *j.startTime_);
    EXPECT_EQ(utc(1, 8, 55), *j.endTime_);
  }

  // Only the ride has to fit: walk 5 min to the vehicle, ride 150 min, walk
  // 60 min after the drop-off (past the end of the drop-off window).
  {
    auto j = itinerary(utc(1, 7, 30), utc(1, 7, 35), api::ModeEnum::WALK);
    j.legs_.push_back(
        itinerary(utc(1, 7, 35), utc(1, 10, 5), api::ModeEnum::FLEX)
            .legs_.front());
    j.legs_.push_back(
        itinerary(utc(1, 10, 5), utc(1, 11, 5), api::ModeEnum::WALK)
            .legs_.front());
    j.endTime_ = utc(1, 11, 5);
    ASSERT_TRUE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 7, 30),
                                            false, j));
    EXPECT_EQ(utc(1, 8, 5), *j.startTime_);
    EXPECT_EQ(utc(1, 8, 10), *j.legs_[1].startTime_);
    EXPECT_EQ(utc(1, 10, 40), *j.legs_[1].endTime_);
    EXPECT_EQ(utc(1, 11, 40), *j.endTime_);
  }

  // Too long to end inside the drop-off window.
  {
    auto j = itinerary(utc(1, 7, 30), utc(1, 10, 30), api::ModeEnum::FLEX);
    EXPECT_FALSE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 7, 30),
                                             false, j));
  }

  // No ride (car_sharing walked all the way): not a flex connection.
  {
    auto j = itinerary(utc(1, 7, 30), utc(1, 7, 45), api::ModeEnum::WALK);
    EXPECT_FALSE(flex::fit_direct_to_windows(*d.tt_, ids, utc(1, 7, 30),
                                             false, j));
  }

  // Not operating on 2019-05-04 (outside the calendar).
  {
    auto j = itinerary(utc(4, 7, 30), utc(4, 7, 45), api::ModeEnum::FLEX);
    EXPECT_FALSE(flex::fit_direct_to_windows(*d.tt_, ids, utc(4, 7, 30),
                                             false, j));
  }
}

// Leg shown with the windows of its own service day. Pickup 08:10-08:40Z.
TEST(motis, flex_service_day) {
  auto const d = load("service_day", "10:10:00,10:40:00");
  auto const id = offer();
  EXPECT_EQ(sys_days{2019_y / May / 1},
            flex::get_service_day(*d.tt_, id, sys_seconds{utc(1, 8, 20)}));
  // 2019-05-02, before that day's window opened: 2019-05-01's window.
  EXPECT_EQ(sys_days{2019_y / May / 1},
            flex::get_service_day(*d.tt_, id, sys_seconds{utc(2, 1, 0)}));
  EXPECT_EQ(sys_days{2019_y / May / 2},
            flex::get_service_day(*d.tt_, id, sys_seconds{utc(2, 8, 15)}));

  auto leg = api::Leg{};
  leg.mode_ = api::ModeEnum::FLEX;
  flex::set_flex_windows(
      *d.tt_, id,
      flex::get_service_day(*d.tt_, id, sys_seconds{utc(2, 1, 0)}), leg);
  EXPECT_EQ(utc(1, 8, 10), **leg.from_.flexStartPickupDropOffWindow_);
  EXPECT_EQ(utc(1, 8, 40), **leg.from_.flexEndPickupDropOffWindow_);
  EXPECT_EQ(utc(1, 8, 0), **leg.to_.flexStartPickupDropOffWindow_);
  EXPECT_EQ(utc(1, 11, 0), **leg.to_.flexEndPickupDropOffWindow_);
}
