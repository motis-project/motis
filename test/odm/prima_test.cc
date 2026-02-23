#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/journey.h"
#include "nigiri/special_stations.h"

#include "motis/odm/odm.h"
#include "motis/odm/prima.h"
#include "motis/transport_mode_ids.h"

#include "motis-api/motis-api.h"

namespace n = nigiri;
namespace nr = nigiri::routing;
using namespace motis::odm;
using namespace std::chrono_literals;
using namespace date;

n::loader::mem_dir tt_files() {
  return n::loader::mem_dir::read(R"__(
"(
# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,A,0.1,0.1,,,,
B,B,B,0.2,0.2,,,,
C,C,C,0.3,0.3,,,,
D,D,D,0.4,0.4,,,,
)__");
}

constexpr auto blacklist_request =
    R"({"start":{"lat":0E0,"lng":0E0},"target":{"lat":0E0,"lng":0E0},"startBusStops":[{"lat":1E-1,"lng":1E-1},{"lat":2E-1,"lng":2E-1}],"targetBusStops":[{"lat":3.0000000000000004E-1,"lng":3.0000000000000004E-1},{"lat":4E-1,"lng":4E-1}],"earliest":0,"latest":172800000,"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

constexpr auto invalid_response = R"({"message":"Internal Error"})";

constexpr auto blacklist_response = R"(
{
  "start": [[{"startTime": 32400000, "endTime": 43200000}],[{"startTime": 43200000, "endTime": 64800000}]],
  "target": [[{"startTime": 43200000, "endTime": 64800000}],[]],
  "direct": [{"startTime": 43200000, "endTime": 64800000}]
}
)";

// 1970-01-01T09:57:00Z, 1970-01-01T10:55:00Z
// 1970-01-01T14:07:00Z, 1970-01-01T14:46:00Z
// 1970-01-01T11:30:00Z, 1970-01-01T12:30:00Z
constexpr auto whitelisting_response = R"(
{
  "start": [[{"pickupTime": 35820000, "dropoffTime": 39300000}],[null]],
  "target": [[{"pickupTime": 50820000, "dropoffTime": 53160000}]],
  "direct": [{"pickupTime": 41400000,"dropoffTime": 45000000}]
}
)";

constexpr auto adjusted_to_whitelisting = R"(
[1970-01-01 09:57, 1970-01-01 12:00]
TRANSFERS: 0
     FROM: (START, START) [1970-01-01 09:57]
       TO: (END, END) [1970-01-01 12:00]
leg 0: (START, START) [1970-01-01 09:57] -> (A, A) [1970-01-01 10:55]
  MUMO (id=16, duration=58)
leg 1: (A, A) [1970-01-01 10:55] -> (A, A) [1970-01-01 11:00]
  FOOTPATH (duration=5)
leg 2: (A, A) [1970-01-01 11:00] -> (END, END) [1970-01-01 12:00]
  MUMO (id=0, duration=60)

[1970-01-01 09:57, 1970-01-01 14:46]
TRANSFERS: 0
     FROM: (START, START) [1970-01-01 09:57]
       TO: (END, END) [1970-01-01 14:46]
leg 0: (START, START) [1970-01-01 09:57] -> (A, A) [1970-01-01 10:55]
  MUMO (id=16, duration=58)
leg 1: (A, A) [1970-01-01 10:55] -> (A, A) [1970-01-01 11:00]
  FOOTPATH (duration=5)
leg 2: (A, A) [1970-01-01 11:00] -> (C, C) [1970-01-01 13:00]
  MUMO (id=1000000, duration=120)
leg 3: (C, C) [1970-01-01 13:00] -> (C, C) [1970-01-01 14:07]
  FOOTPATH (duration=67)
leg 4: (C, C) [1970-01-01 14:07] -> (END, END) [1970-01-01 14:46]
  MUMO (id=16, duration=39)

)";

TEST(odm, prima_update) {
  n::timetable tt;
  tt.date_range_ = {date::sys_days{2017_y / January / 1},
                    date::sys_days{2017_y / January / 2}};
  n::loader::register_special_stations(tt);
  auto const src = n::source_idx_t{0};
  n::loader::gtfs::load_timetable({.default_tz_ = "Europe/Berlin"}, src,
                                  tt_files(), tt);
  n::loader::finalize(tt);

  auto const get_loc_idx = [&](auto&& s) {
    return tt.locations_.location_id_to_idx_.at({.id_ = s, .src_ = src});
  };

  auto const loc = osr::location{};
  auto p = prima{"prima_url", loc, loc, motis::api::plan_params{}};
  p.fixed_ = n::event_type::kDep;
  p.cap_ = {.wheelchairs_ = 1, .bikes_ = 0, .passengers_ = 1, .luggage_ = 0};
  p.first_mile_taxi_ = {
      {get_loc_idx("A"), n::duration_t{60min}, motis::kOdmTransportModeId},
      {get_loc_idx("B"), n::duration_t{60min}, motis::kOdmTransportModeId}};
  p.last_mile_taxi_ = {
      {get_loc_idx("C"), n::duration_t{60min}, motis::kOdmTransportModeId},
      {get_loc_idx("D"), n::duration_t{60min}, motis::kOdmTransportModeId}};

  EXPECT_EQ(p.make_blacklist_taxi_request(
                tt, {n::unixtime_t{0h}, n::unixtime_t{48h}}),
            blacklist_request);

  EXPECT_FALSE(p.consume_blacklist_taxi_response(invalid_response));
  EXPECT_TRUE(p.consume_blacklist_taxi_response(blacklist_response));

  ASSERT_EQ(p.first_mile_taxi_.size(), 2);
  EXPECT_EQ(p.first_mile_taxi_[0].target_, get_loc_idx("A"));
  ASSERT_EQ(p.first_mile_taxi_times_[0].size(), 1);
  EXPECT_EQ(p.first_mile_taxi_times_[0][0].from_, to_unix(32400000));
  EXPECT_EQ(p.first_mile_taxi_times_[0][0].to_, to_unix(43200000));
  EXPECT_EQ(p.first_mile_taxi_[1].target_, get_loc_idx("B"));
  ASSERT_EQ(p.first_mile_taxi_times_[1].size(), 1);
  EXPECT_EQ(p.first_mile_taxi_times_[1][0].from_, to_unix(43200000));
  EXPECT_EQ(p.first_mile_taxi_times_[1][0].to_, to_unix(64800000));

  ASSERT_EQ(p.last_mile_taxi_.size(), 2);
  EXPECT_EQ(p.last_mile_taxi_[0].target_, get_loc_idx("C"));
  ASSERT_EQ(p.last_mile_taxi_times_[0].size(), 1);
  EXPECT_EQ(p.last_mile_taxi_times_[0][0].from_, to_unix(43200000));
  EXPECT_EQ(p.last_mile_taxi_times_[0][0].to_, to_unix(64800000));
  EXPECT_EQ(p.last_mile_taxi_[1].target_, get_loc_idx("D"));
  EXPECT_EQ(p.last_mile_taxi_times_[1].size(), 0);

  auto const expected_direct_interval =
      n::interval{to_unix(43200000), to_unix(64800000)};
  for (auto const& d : p.direct_taxi_) {
    EXPECT_TRUE(expected_direct_interval.contains(d.dep_));
  }

  auto taxi_journeys = std::vector<nr::journey>{};
  taxi_journeys.push_back(
      {.legs_ = {{n::direction::kForward,
                  n::get_special_station(n::special_station::kStart),
                  get_loc_idx("A"), n::unixtime_t{10h}, n::unixtime_t{11h},
                  nr::offset{get_loc_idx("A"), 1h, motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("A"),
                  n::get_special_station(n::special_station::kEnd),
                  n::unixtime_t{11h}, n::unixtime_t{12h},
                  nr::offset{get_loc_idx("A"), 1h, kWalkTransportModeId}}},
       .start_time_ = n::unixtime_t{10h},
       .dest_time_ = n::unixtime_t{12h},
       .dest_ = n::get_special_station(n::special_station::kEnd)});

  taxi_journeys.push_back(
      {.legs_ = {{n::direction::kForward,
                  n::get_special_station(n::special_station::kStart),
                  get_loc_idx("B"), n::unixtime_t{11h}, n::unixtime_t{12h},
                  nr::offset{get_loc_idx("B"), 1h, motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("B"),
                  n::get_special_station(n::special_station::kEnd),
                  n::unixtime_t{12h}, n::unixtime_t{13h},
                  nr::offset{get_loc_idx("B"), 1h, kWalkTransportModeId}}},
       .start_time_ = n::unixtime_t{11h},
       .dest_time_ = n::unixtime_t{13h},
       .dest_ = n::get_special_station(n::special_station::kEnd)});

  taxi_journeys.push_back(
      {.legs_ = {{n::direction::kForward,
                  n::get_special_station(n::special_station::kStart),
                  get_loc_idx("A"), n::unixtime_t{10h}, n::unixtime_t{11h},
                  n::routing::offset{get_loc_idx("A"), 1h,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("A"), get_loc_idx("C"),
                  n::unixtime_t{11h}, n::unixtime_t{13h},
                  nr::offset{get_loc_idx("C"), 2h, motis::kFlexModeIdOffset}},
                 {n::direction::kForward, get_loc_idx("C"),
                  n::get_special_station(n::special_station::kEnd),
                  n::unixtime_t{13h}, n::unixtime_t{14h},
                  nr::offset{get_loc_idx("C"), 1h,
                             motis::kOdmTransportModeId}}},
       .start_time_ = n::unixtime_t{10h},
       .dest_time_ = n::unixtime_t{14h},
       .dest_ = n::get_special_station(n::special_station::kEnd)});

  p.direct_taxi_ = {
      direct_ride{.dep_ = n::unixtime_t{11h}, .arr_ = n::unixtime_t{12h}}};

  auto first_mile_taxi_rides = std::vector<nr::start>{};
  auto last_mile_taxi_rides = std::vector<nr::start>{};
  extract_taxis(taxi_journeys, first_mile_taxi_rides, last_mile_taxi_rides);
  EXPECT_FALSE(p.consume_whitelist_taxi_response(
      invalid_response, taxi_journeys, first_mile_taxi_rides,
      last_mile_taxi_rides));
  EXPECT_TRUE(p.consume_whitelist_taxi_response(
      whitelisting_response, taxi_journeys, first_mile_taxi_rides,
      last_mile_taxi_rides));

  auto ss = std::stringstream{};
  ss << "\n";
  for (auto const& j : taxi_journeys) {
    j.print(ss, tt, nullptr);
    ss << "\n";
  }

  EXPECT_EQ(adjusted_to_whitelisting, ss.str());
}
