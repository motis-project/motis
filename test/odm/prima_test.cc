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

constexpr auto const invalid_response = R"({"message":"Internal Error"})";

// constexpr auto const blacklisting_response = R"(
// {
//   "start": [[true,null],[true]],
//   "target": [[true],[false]],
//   "direct": [false,true]
// }
// )";

constexpr auto const blacklisting_response = R"(
{
  "start": [[{"start": 32400000, "end": 43200000}],[{"start": 43200000, "end": 64800000}]],
  "target": [[{"start": 43200000, "end": 64800000}],[]],
  "direct": [{"start": 43200000, "end": 64800000}]
}
)";

constexpr auto const blacklisted =
    R"({"start":{"lat":0E0,"lng":0E0},"target":{"lat":0E0,"lng":0E0},"startBusStops":[{"lat":1E-1,"lng":1E-1,"times":[39300000]},{"lat":2E-1,"lng":2E-1,"times":[42900000]}],"targetBusStops":[{"lat":3.0000000000000004E-1,"lng":3.0000000000000004E-1,"times":[47100000]}],"directTimes":[39600000],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

// 1970-01-01T09:57:00Z, 1970-01-01T10:55:00Z
// 1970-01-01T14:07:00Z, 1970-01-01T14:46:00Z
// 1970-01-01T11:30:00Z, 1970-01-01T12:30:00Z
constexpr auto const whitelisting_response = R"(
{
  "start": [[{"pickupTime": 35820000, "dropoffTime": 39300000}],[null]],
  "target": [[{"pickupTime": 50820000, "dropoffTime": 53160000}]],
  "direct": [{"pickupTime": 41400000,"dropoffTime": 45000000}]
}
)";

constexpr auto const adjusted_to_whitelisting = R"(
[1970-01-01 09:57, 1970-01-01 12:00]
TRANSFERS: 0
     FROM: (START, START) [1970-01-01 09:57]
       TO: (END, END) [1970-01-01 12:00]
leg 0: (START, START) [1970-01-01 09:57] -> (A, A) [1970-01-01 10:55]
  MUMO (id=10, duration=58)
leg 1: (A, A) [1970-01-01 10:55] -> (A, A) [1970-01-01 11:00]
  FOOTPATH (duration=5)
leg 2: (A, A) [1970-01-01 11:00] -> (END, END) [1970-01-01 12:00]
  MUMO (id=0, duration=60)

[1970-01-01 09:57, 1970-01-01 14:46]
TRANSFERS: 0
     FROM: (START, START) [1970-01-01 09:57]
       TO: (END, END) [1970-01-01 14:46]
leg 0: (START, START) [1970-01-01 09:57] -> (A, A) [1970-01-01 10:55]
  MUMO (id=10, duration=58)
leg 1: (A, A) [1970-01-01 10:55] -> (A, A) [1970-01-01 11:00]
  FOOTPATH (duration=5)
leg 2: (A, A) [1970-01-01 11:00] -> (C, C) [1970-01-01 13:00]
  FOOTPATH (duration=120)
leg 3: (C, C) [1970-01-01 13:00] -> (C, C) [1970-01-01 14:07]
  FOOTPATH (duration=67)
leg 4: (C, C) [1970-01-01 14:07] -> (END, END) [1970-01-01 14:46]
  MUMO (id=10, duration=39)

)";

TEST(odm, prima_update) {
  using namespace nigiri;
  using namespace nigiri::loader;
  using namespace nigiri::loader::gtfs;

  timetable tt;
  tt.date_range_ = {date::sys_days{2017_y / January / 1},
                    date::sys_days{2017_y / January / 2}};
  register_special_stations(tt);
  auto const src = source_idx_t{0};
  gtfs::load_timetable({}, src, tt_files(), tt);
  finalize(tt);

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
  p.direct_taxi_ = {};

  EXPECT_FALSE(p.consume_blacklist_taxis_response(invalid_response));
  EXPECT_FALSE(p.consume_blacklist_taxis_response(blacklisting_response));

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
      n::interval<n::unixtime_t>{to_unix(43200000), to_unix(64800000)};
  for (auto const& d : p.direct_taxi_) {
    EXPECT_TRUE(expected_direct_interval.contains(d.dep_));
  }

  auto taxi_journeys = std::vector<nigiri::routing::journey>{};
  taxi_journeys.push_back(
      {.legs_ = {{n::direction::kForward,
                  get_special_station(special_station::kStart),
                  get_loc_idx("A"), n::unixtime_t{10h}, n::unixtime_t{11h},
                  n::routing::offset{get_loc_idx("A"), 1h,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("A"),
                  get_special_station(special_station::kEnd),
                  n::unixtime_t{11h}, n::unixtime_t{12h},
                  n::routing::offset{get_loc_idx("A"), 1h,
                                     kWalkTransportModeId}}},
       .start_time_ = n::unixtime_t{10h},
       .dest_time_ = n::unixtime_t{12h},
       .dest_ = get_special_station(special_station::kEnd)});

  taxi_journeys.push_back(
      {.legs_ = {{n::direction::kForward,
                  get_special_station(special_station::kStart),
                  get_loc_idx("B"), n::unixtime_t{11h}, n::unixtime_t{12h},
                  n::routing::offset{get_loc_idx("B"), 1h,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("B"),
                  get_special_station(special_station::kEnd),
                  n::unixtime_t{12h}, n::unixtime_t{13h},
                  n::routing::offset{get_loc_idx("B"), 1h,
                                     kWalkTransportModeId}}},
       .start_time_ = n::unixtime_t{11h},
       .dest_time_ = n::unixtime_t{13h},
       .dest_ = get_special_station(special_station::kEnd)});

  taxi_journeys.push_back(
      {.legs_ = {{n::direction::kForward,
                  get_special_station(special_station::kStart),
                  get_loc_idx("A"), n::unixtime_t{10h}, n::unixtime_t{11h},
                  n::routing::offset{get_loc_idx("A"), 1h,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("A"), get_loc_idx("C"),
                  n::unixtime_t{11h}, n::unixtime_t{13h},
                  n::footpath{get_loc_idx("C"), 2h}},
                 {n::direction::kForward, get_loc_idx("C"),
                  get_special_station(special_station::kEnd),
                  n::unixtime_t{13h}, n::unixtime_t{14h},
                  n::routing::offset{get_loc_idx("C"), 1h,
                                     motis::kOdmTransportModeId}}},
       .start_time_ = n::unixtime_t{10h},
       .dest_time_ = n::unixtime_t{14h},
       .dest_ = get_special_station(special_station::kEnd)});

  EXPECT_FALSE(
      p.consume_whitelist_taxis_response(invalid_response, taxi_journeys));
  EXPECT_TRUE(
      p.consume_whitelist_taxis_response(whitelisting_response, taxi_journeys));

  auto ss = std::stringstream{};
  ss << "\n";
  for (auto const& j : taxi_journeys) {
    j.print(ss, tt, nullptr);
    ss << "\n";
  }

  EXPECT_EQ(adjusted_to_whitelisting, ss.str());
}
