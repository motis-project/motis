#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/special_stations.h"

#include "motis/odm/equal_journeys.h"
#include "motis/odm/mixer.h"
#include "motis/odm/odm.h"
#include "motis/odm/prima.h"

namespace n = nigiri;
using namespace motis::odm;
using namespace std::chrono_literals;

static auto const kOdmMixer = mixer{.walk_cost_ = {{0, 1}, {15, 11}},
                                    .taxi_cost_ = {{0, 59}, {1, 13}},
                                    .transfer_cost_ = {{0, 15}},
                                    .direct_taxi_factor_ = 1.3,
                                    .direct_taxi_constant_ = 27,
                                    .travel_time_weight_ = 1.5,
                                    .distance_weight_ = 0.07,
                                    .distance_exponent_ = 1.5};

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
                                 arr - dep, kODM}}},
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
      .legs_ = {n::routing::journey::leg{
          n::direction::kForward,
          get_special_station(n::special_station::kStart),
          n::location_idx_t{23U}, n::unixtime_t{10h + 43min},
          n::unixtime_t{10h + 47min},
          n::routing::offset{n::location_idx_t{23U}, 4min, kODM}}},
      .start_time_ = n::unixtime_t{10h + 43min},
      .dest_time_ = n::unixtime_t{11h},
      .dest_ = get_special_station(n::special_station::kEnd),
      .transfers_ = 0U};

  auto odm_journeys = std::vector<n::routing::journey>{
      pt_taxi,
      direct_taxi(n::unixtime_t{10h + 17min}, n::unixtime_t{10h + 27min}),
      direct_taxi(n::unixtime_t{10h + 43min}, n::unixtime_t{10h + 53min}),
      direct_taxi(n::unixtime_t{10h + 50min}, n::unixtime_t{11h + 00min}),
      direct_taxi(n::unixtime_t{10h + 00min}, n::unixtime_t{10h + 10min}),
      direct_taxi(n::unixtime_t{11h + 00min}, n::unixtime_t{10h + 10min})};

  kOdmMixer.mix(pt_journeys, odm_journeys);

  ASSERT_EQ(odm_journeys.size(), 2U);
  EXPECT_NE(std::find_if(begin(odm_journeys), end(odm_journeys),
                         [&](auto const& j) { return j == pt; }),
            end(odm_journeys));
  EXPECT_NE(std::find_if(begin(odm_journeys), end(odm_journeys),
                         [&](auto const& j) { return j == pt_taxi; }),
            end(odm_journeys));
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
                  n::routing::offset{n::location_idx_t{24U}, 6min, kODM}},
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
                  n::routing::offset{n::location_idx_t{25U}, 10min, kODM}},
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
                  n::routing::offset{n::location_idx_t{26U}, 15min, kODM}},
                 {n::direction::kForward, n::location_idx_t{42U},
                  get_special_station(n::special_station::kEnd),
                  n::unixtime_t{10h + 55min}, n::unixtime_t{11h},
                  n::routing::offset{n::location_idx_t{42U}, 5min, kWalk}}},
       .start_time_ = n::unixtime_t{10h + 30min},
       .dest_time_ = n::unixtime_t{11h},
       .dest_ = get_special_station(n::special_station::kEnd),
       .transfers_ = 0U}};

  kOdmMixer.mix(pt_journeys, odm_journeys);

  ASSERT_EQ(odm_journeys.size(), 1U);
  EXPECT_NE(std::find_if(begin(odm_journeys), end(odm_journeys),
                         [&](auto const& j) { return j == pt; }),
            end(odm_journeys));
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

constexpr auto const initial =
    R"({"start":{"lat":0E0,"lng":0E0},"target":{"lat":1E0,"lng":1E0},"startBusStops":[{"coordinates":{"lat":1E-1,"lng":1E-1},"times":["1970-01-01T10:55:00Z","1970-01-01T11:55:00Z"]},{"coordinates":{"lat":2E-1,"lng":2E-1},"times":["1970-01-01T11:55:00Z"]}],"targetBusStops":[{"coordinates":{"lat":3.0000000000000004E-1,"lng":3.0000000000000004E-1},"times":["1970-01-01T13:05:00Z"]},{"coordinates":{"lat":4E-1,"lng":4E-1},"times":["1970-01-01T14:05:00Z"]}],"times":["1970-01-01T10:00:00Z","1970-01-01T11:00:00Z"],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

constexpr auto const invalid_response = R"({"message":"Internal Error"})";

constexpr auto const blacklisting_response = R"(
{
  "start": [[true,false],[true]],
  "target": [[true],[false]],
  "direct": [false,true]
}
)";

constexpr auto const blacklisted =
    R"({"start":{"lat":0E0,"lng":0E0},"target":{"lat":1E0,"lng":1E0},"startBusStops":[{"coordinates":{"lat":1E-1,"lng":1E-1},"times":["1970-01-01T10:55:00Z"]},{"coordinates":{"lat":2E-1,"lng":2E-1},"times":["1970-01-01T11:55:00Z"]}],"targetBusStops":[{"coordinates":{"lat":3.0000000000000004E-1,"lng":3.0000000000000004E-1},"times":["1970-01-01T13:05:00Z"]}],"times":["1970-01-01T11:00:00Z"],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

constexpr auto const whitelisting_response = R"(
{
  "start": [["1970-01-01T09:45:00Z"],[null]],
  "target": [["1970-01-01T14:05:00Z"]],
  "direct": ["1970-01-01T12:30:00Z"]
}
)";

constexpr auto const adjusted_to_whitelisting = R"(
[1970-01-01 09:45, 1970-01-01 12:00]
TRANSFERS: 0
     FROM: (START, START) [1970-01-01 09:45]
       TO: (END, END) [1970-01-01 12:00]
leg 0: (START, START) [1970-01-01 09:45] -> (A, A) [1970-01-01 10:55]
  MUMO (id=5, duration=70)
leg 1: (A, A) [1970-01-01 10:55] -> (A, A) [1970-01-01 11:00]
  FOOTPATH (duration=5)
leg 2: (A, A) [1970-01-01 11:00] -> (END, END) [1970-01-01 12:00]
  MUMO (id=0, duration=60)

[1970-01-01 09:45, 1970-01-01 14:05]
TRANSFERS: 0
     FROM: (START, START) [1970-01-01 09:45]
       TO: (END, END) [1970-01-01 14:05]
leg 0: (START, START) [1970-01-01 09:45] -> (A, A) [1970-01-01 10:55]
  MUMO (id=5, duration=70)
leg 1: (A, A) [1970-01-01 10:55] -> (A, A) [1970-01-01 11:00]
  FOOTPATH (duration=5)
leg 2: (A, A) [1970-01-01 11:00] -> (C, C) [1970-01-01 13:00]
  FOOTPATH (duration=120)
leg 3: (C, C) [1970-01-01 13:00] -> (C, C) [1970-01-01 13:05]
  FOOTPATH (duration=5)
leg 4: (C, C) [1970-01-01 13:05] -> (END, END) [1970-01-01 14:05]
  MUMO (id=5, duration=60)

)";

TEST(odm, prima_update) {
  using namespace nigiri;
  using namespace nigiri::loader;
  using namespace nigiri::loader::gtfs;

  timetable tt;
  tt.date_range_ = {2017y / std::chrono::January / 1,
                    2017y / std::chrono::January / 2};
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
      .fixed_ = fixed::kDep,
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

  EXPECT_EQ(initial, p.get_msg_str(tt));
  EXPECT_FALSE(p.blacklist_update(invalid_response));
  p.blacklist_update(blacklisting_response);
  EXPECT_EQ(blacklisted, p.get_msg_str(tt));
  EXPECT_FALSE(p.whitelist_update(invalid_response));
  p.whitelist_update(whitelisting_response);
  EXPECT_EQ(blacklisted, p.get_msg_str(tt));

  p.odm_journeys_.push_back(
      {.legs_ = {{n::direction::kForward,
                  get_special_station(special_station::kStart),
                  get_loc_idx("A"), n::unixtime_t{10h}, n::unixtime_t{11h},
                  n::routing::offset{get_loc_idx("A"), 1h, kODM}},
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
                  n::routing::offset{get_loc_idx("B"), 1h, kODM}},
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
                  n::routing::offset{get_loc_idx("A"), 1h, kODM}},
                 {n::direction::kForward, get_loc_idx("A"), get_loc_idx("C"),
                  n::unixtime_t{11h}, n::unixtime_t{13h},
                  n::footpath{get_loc_idx("C"), 2h}},
                 {n::direction::kForward, get_loc_idx("C"),
                  get_special_station(special_station::kEnd),
                  n::unixtime_t{13h}, n::unixtime_t{14h},
                  n::routing::offset{get_loc_idx("C"), 1h, kODM}}},
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