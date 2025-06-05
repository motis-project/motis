#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/special_stations.h"

#include "motis/odm/mixer.h"
#include "motis/odm/odm.h"
#include "motis/odm/prima.h"
#include "motis/transport_mode_ids.h"

namespace n = nigiri;
using namespace motis::odm;
using namespace std::chrono_literals;
using namespace date;

TEST(odm, tally) {
  auto const ct = std::vector<cost_threshold>{{0, 30}, {1, 1}, {10, 2}};
  EXPECT_EQ(0, tally(0, ct));
  EXPECT_EQ(30, tally(1, ct));
  EXPECT_EQ(43, tally(12, ct));
}

n::routing::journey direct_taxi(n::unixtime_t const dep,
                                n::unixtime_t const arr) {
  return {.legs_ = {n::routing::journey::leg{
              n::direction::kForward,
              get_special_station(n::special_station::kStart),
              get_special_station(n::special_station::kEnd), dep, arr,
              n::routing::offset{get_special_station(n::special_station::kEnd),
                                 arr - dep, motis::kOdmTransportModeId}}},
          .start_time_ = dep,
          .dest_time_ = arr,
          .dest_ = get_special_station(n::special_station::kEnd),
          .transfers_ = 0U};
}

TEST(odm, pt_taxi_no_direct) {
  auto pt = n::routing::journey{
      .legs_ = {n::routing::journey::leg{
          n::direction::kForward,
          get_special_station(n::special_station::kStart),
          n::location_idx_t{23U}, n::unixtime_t{10h + 17min},
          n::unixtime_t{10h + 47min},
          n::routing::offset{n::location_idx_t{23U}, 30min, kWalk}}},
      .start_time_ = n::unixtime_t{10h + 17min},
      .dest_time_ = n::unixtime_t{11h},
      .dest_ = get_special_station(n::special_station::kEnd),
      .transfers_ = 0U};

  auto pt_journeys = n::pareto_set<n::routing::journey>{};
  pt_journeys.add(n::routing::journey{pt});

  auto pt_taxi = n::routing::journey{
      .legs_ = {{n::direction::kForward,
                 get_special_station(n::special_station::kStart),
                 n::location_idx_t{23U}, n::unixtime_t{10h + 43min},
                 n::unixtime_t{10h + 47min},
                 n::routing::offset{n::location_idx_t{23U}, 4min,
                                    motis::kOdmTransportModeId}},
                {n::direction::kForward, n::location_idx_t{23U},
                 get_special_station(n::special_station::kEnd),
                 n::unixtime_t{10h + 47min}, n::unixtime_t{11h},
                 n::routing::journey::run_enter_exit{
                     n::rt::run{}, n::stop_idx_t{0}, n::stop_idx_t{1}}}},
      .start_time_ = n::unixtime_t{10h + 43min},
      .dest_time_ = n::unixtime_t{11h},
      .dest_ = get_special_station(n::special_station::kEnd),
      .transfers_ = 0U};

  auto odm_journeys = std::vector<n::routing::journey>{
      pt_taxi,
      direct_taxi(n::unixtime_t{10h + 10min}, n::unixtime_t{10h + 20min}),
      direct_taxi(n::unixtime_t{10h + 17min}, n::unixtime_t{10h + 27min}),
      direct_taxi(n::unixtime_t{10h + 43min}, n::unixtime_t{10h + 53min}),
      direct_taxi(n::unixtime_t{10h + 50min}, n::unixtime_t{11h + 00min}),
      direct_taxi(n::unixtime_t{11h + 00min}, n::unixtime_t{11h + 10min})};

  get_default_mixer().mix(pt_journeys, odm_journeys, nullptr);

  ASSERT_EQ(odm_journeys.size(), 2U);
  EXPECT_NE(utl::find(odm_journeys, pt), end(odm_journeys));
  EXPECT_NE(utl::find(odm_journeys, pt_taxi), end(odm_journeys));
}

TEST(odm, taxi_saves_transfers) {
  auto pt = n::routing::journey{
      .legs_ = {n::routing::journey::leg{
                    n::direction::kForward,
                    get_special_station(n::special_station::kStart),
                    n::location_idx_t{23U}, n::unixtime_t{10h},
                    n::unixtime_t{10h + 5min},
                    n::routing::offset{n::location_idx_t{23U}, 5min, kWalk}},
                n::routing::journey::leg{
                    n::direction::kForward, n::location_idx_t{42U},
                    get_special_station(n::special_station::kEnd),
                    n::unixtime_t{10h + 55min}, n::unixtime_t{11h},
                    n::routing::offset{n::location_idx_t{42U}, 5min, kWalk}}},
      .start_time_ = n::unixtime_t{10h},
      .dest_time_ = n::unixtime_t{11h},
      .dest_ = get_special_station(n::special_station::kEnd),
      .transfers_ = 4U};

  auto pt_journeys = n::pareto_set<n::routing::journey>{};
  pt_journeys.add(n::routing::journey{pt});

  auto odm_journeys = std::vector<n::routing::journey>{
      {.legs_ = {{n::direction::kForward,
                  get_special_station(n::special_station::kStart),
                  n::location_idx_t{24U}, n::unixtime_t{10h + 14min},
                  n::unixtime_t{10h + 20min},
                  n::routing::offset{n::location_idx_t{24U}, 6min,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, n::location_idx_t{42U},
                  get_special_station(n::special_station::kEnd),
                  n::unixtime_t{10h + 55min}, n::unixtime_t{11h},
                  n::routing::offset{n::location_idx_t{42U}, 5min, kWalk}}},
       .start_time_ = n::unixtime_t{10h + 14min},
       .dest_time_ = n::unixtime_t{11h},
       .dest_ = get_special_station(n::special_station::kEnd),
       .transfers_ = 2U},
      {.legs_ = {{n::direction::kForward,
                  get_special_station(n::special_station::kStart),
                  n::location_idx_t{25U}, n::unixtime_t{10h + 20min},
                  n::unixtime_t{10h + 30min},
                  n::routing::offset{n::location_idx_t{25U}, 10min,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, n::location_idx_t{42U},
                  get_special_station(n::special_station::kEnd),
                  n::unixtime_t{10h + 55min}, n::unixtime_t{11h},
                  n::routing::offset{n::location_idx_t{42U}, 5min, kWalk}}},
       .start_time_ = n::unixtime_t{10h + 20min},
       .dest_time_ = n::unixtime_t{11h},
       .dest_ = get_special_station(n::special_station::kEnd),
       .transfers_ = 1U},
      {.legs_ = {{n::direction::kForward,
                  get_special_station(n::special_station::kStart),
                  n::location_idx_t{26U}, n::unixtime_t{10h + 30min},
                  n::unixtime_t{10h + 45min},
                  n::routing::offset{n::location_idx_t{26U}, 15min,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, n::location_idx_t{42U},
                  get_special_station(n::special_station::kEnd),
                  n::unixtime_t{10h + 55min}, n::unixtime_t{11h},
                  n::routing::offset{n::location_idx_t{42U}, 5min, kWalk}}},
       .start_time_ = n::unixtime_t{10h + 30min},
       .dest_time_ = n::unixtime_t{11h},
       .dest_ = get_special_station(n::special_station::kEnd),
       .transfers_ = 0U}};

  get_default_mixer().mix(pt_journeys, odm_journeys, nullptr);

  ASSERT_EQ(odm_journeys.size(), 1U);
  EXPECT_NE(utl::find(odm_journeys, pt), end(odm_journeys));
}

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

constexpr auto const kExpectedInitial =
    R"({"start":{"lat":0E0,"lng":0E0},"target":{"lat":1E0,"lng":1E0},"startBusStops":[{"lat":1E-1,"lng":1E-1,"times":[39300000,42900000]},{"lat":2E-1,"lng":2E-1,"times":[42900000]}],"targetBusStops":[{"lat":3.0000000000000004E-1,"lng":3.0000000000000004E-1,"times":[47100000]},{"lat":4E-1,"lng":4E-1,"times":[50700000]}],"directTimes":[36000000,39600000],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

constexpr auto const invalid_response = R"({"message":"Internal Error"})";

constexpr auto const blacklisting_response = R"(
{
  "start": [[true,null],[true]],
  "target": [[true],[false]],
  "direct": [false,true]
}
)";

constexpr auto const blacklisted =
    R"({"start":{"lat":0E0,"lng":0E0},"target":{"lat":1E0,"lng":1E0},"startBusStops":[{"lat":1E-1,"lng":1E-1,"times":[39300000]},{"lat":2E-1,"lng":2E-1,"times":[42900000]}],"targetBusStops":[{"lat":3.0000000000000004E-1,"lng":3.0000000000000004E-1,"times":[47100000]}],"directTimes":[39600000],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

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

  auto p = prima{
      .from_ = {0.0, 0.0},
      .to_ = {1.0, 1.0},
      .fixed_ = n::event_type::kDep,
      .cap_ = {.wheelchairs_ = 1, .bikes_ = 0, .passengers_ = 1, .luggage_ = 0},
      .from_rides_ = {{.time_at_start_ = n::unixtime_t{10h},
                       .time_at_stop_ = n::unixtime_t{11h},
                       .stop_ = get_loc_idx("A")},
                      {.time_at_start_ = n::unixtime_t{11h},
                       .time_at_stop_ = n::unixtime_t{12h},
                       .stop_ = get_loc_idx("A")},
                      {.time_at_start_ = n::unixtime_t{11h},
                       .time_at_stop_ = n::unixtime_t{12h},
                       .stop_ = get_loc_idx("B")}},
      .to_rides_ = {{.time_at_start_ = n::unixtime_t{14h},
                     .time_at_stop_ = n::unixtime_t{13h},
                     .stop_ = get_loc_idx("C")},
                    {.time_at_start_ = n::unixtime_t{15h},
                     .time_at_stop_ = n::unixtime_t{14h},
                     .stop_ = get_loc_idx("D")}},
      .direct_rides_ = {
          {.dep_ = n::unixtime_t{10h}, .arr_ = n::unixtime_t{11h}},
          {.dep_ = n::unixtime_t{11h}, .arr_ = n::unixtime_t{12h}}}};

  EXPECT_EQ(kExpectedInitial, p.get_prima_request(tt));
  EXPECT_FALSE(p.blacklist_update(invalid_response));
  EXPECT_TRUE(p.blacklist_update(blacklisting_response));
  EXPECT_EQ(blacklisted, p.get_prima_request(tt));
  EXPECT_FALSE(p.whitelist_update(invalid_response));
  EXPECT_TRUE(p.whitelist_update(whitelisting_response));

  p.odm_journeys_.push_back(
      {.legs_ = {{n::direction::kForward,
                  get_special_station(special_station::kStart),
                  get_loc_idx("A"), n::unixtime_t{10h}, n::unixtime_t{11h},
                  n::routing::offset{get_loc_idx("A"), 1h,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("A"),
                  get_special_station(special_station::kEnd),
                  n::unixtime_t{11h}, n::unixtime_t{12h},
                  n::routing::offset{get_loc_idx("A"), 1h, kWalk}}},
       .start_time_ = n::unixtime_t{10h},
       .dest_time_ = n::unixtime_t{12h},
       .dest_ = get_special_station(special_station::kEnd)});

  p.odm_journeys_.push_back(
      {.legs_ = {{n::direction::kForward,
                  get_special_station(special_station::kStart),
                  get_loc_idx("B"), n::unixtime_t{11h}, n::unixtime_t{12h},
                  n::routing::offset{get_loc_idx("B"), 1h,
                                     motis::kOdmTransportModeId}},
                 {n::direction::kForward, get_loc_idx("B"),
                  get_special_station(special_station::kEnd),
                  n::unixtime_t{12h}, n::unixtime_t{13h},
                  n::routing::offset{get_loc_idx("B"), 1h, kWalk}}},
       .start_time_ = n::unixtime_t{11h},
       .dest_time_ = n::unixtime_t{13h},
       .dest_ = get_special_station(special_station::kEnd)});

  p.odm_journeys_.push_back(
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

  p.adjust_to_whitelisting();

  auto ss = std::stringstream{};
  ss << "\n";
  for (auto const& j : p.odm_journeys_) {
    j.print(ss, tt, nullptr);
    ss << "\n";
  }

  EXPECT_EQ(adjusted_to_whitelisting, ss.str());
}
