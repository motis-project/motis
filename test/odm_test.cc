#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/special_stations.h"

#include "motis/odm/calibration/json.h"
#include "motis/odm/equal_journeys.h"
#include "motis/odm/mix.h"
#include "motis/odm/odm.h"
#include "motis/odm/prima_state.h"

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

TEST(mix, tally) {
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

TEST(mix, pt_taxi_no_direct) {
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

TEST(mix, taxi_saves_transfers) {
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

constexpr auto const parameters_json = R"(
{
  "params": {
    "weightTravelTime": 1.5,
    "weightTimeDistance": 0.07,
    "exponentTimeDistance": 1.5,
    "costWalk": [
      { "threshold": 0, "cost": 1 },
      { "threshold": 15, "cost": 11 }
    ],
    "costTaxi": [
      { "threshold": 0, "cost": 59 },
      { "threshold": 1, "cost": 13 }
    ],
    "costTransfer": [{ "threshold": 0, "cost": 15 }],
    "factorDirectTaxi": 1.3,
    "constantDirectTaxi": 27
  }
}
)";

TEST(odm_calibration, read_parameters) {
  auto const m = read_parameters(parameters_json);
  EXPECT_EQ(m.travel_time_weight_, 1.5);
  EXPECT_EQ(m.distance_weight_, 0.07);
  EXPECT_EQ(m.distance_exponent_, 1.5);
  ASSERT_EQ(m.walk_cost_.size(), 2);
  EXPECT_EQ(m.walk_cost_[0].threshold_, 0);
  EXPECT_EQ(m.walk_cost_[0].cost_, 1);
  EXPECT_EQ(m.walk_cost_[1].threshold_, 15);
  EXPECT_EQ(m.walk_cost_[1].cost_, 11);
  ASSERT_EQ(m.taxi_cost_.size(), 2);
  EXPECT_EQ(m.taxi_cost_[0].threshold_, 0);
  EXPECT_EQ(m.taxi_cost_[0].cost_, 59);
  EXPECT_EQ(m.taxi_cost_[1].threshold_, 1);
  EXPECT_EQ(m.taxi_cost_[1].cost_, 13);
  ASSERT_EQ(m.transfer_cost_.size(), 1);
  EXPECT_EQ(m.transfer_cost_[0].threshold_, 0);
  EXPECT_EQ(m.transfer_cost_[0].cost_, 15);
  EXPECT_EQ(m.direct_taxi_factor_, 1.3);
  EXPECT_EQ(m.direct_taxi_constant_, 27);
}

constexpr auto const requirements_json = R"(
{
    "conSets": [
        [
            {
                "name": "ÖV",
                "departure": "10:17",
                "arrival": "11:00",
                "transfers": 0,
                "startMode": "walk",
                "startLength": 30,
                "endMode": "walk",
                "endLength": 0,
                "toDom": ""
            },
            {
                "name": "ÖV+Taxi",
                "departure": "10:43",
                "arrival": "11:00",
                "transfers": 0,
                "startMode": "taxi",
                "startLength": 4,
                "endMode": "walk",
                "endLength": 0,
                "toDom": ""
            },
            {
                "name": "Direkt-Taxi Überholt 1",
                "departure": "10:17",
                "arrival": "10:27",
                "transfers": 0,
                "startMode": "taxi",
                "startLength": 10,
                "endMode": "walk",
                "endLength": 0,
                "toDom": true
            }
        ],
        [
            {
                "name": "ÖV",
                "departure": "10:00",
                "arrival": "11:00",
                "transfers": 4,
                "startMode": "walk",
                "startLength": 5,
                "endMode": "walk",
                "endLength": 5,
                "toDom": ""
            },
            {
                "name": "ÖV+Taxi 1",
                "departure": "10:14",
                "arrival": "11:00",
                "transfers": 2,
                "startMode": "taxi",
                "startLength": 6,
                "endMode": "walk",
                "endLength": 5,
                "toDom": true
            }
        ]
    ]
}
)";

TEST(odm_calibration, read_requirements) {
  auto const reqs = read_requirements(requirements_json);

  ASSERT_EQ(reqs.size(), 2);
  ASSERT_EQ(reqs[0].pt_.size(), 1);
  ASSERT_EQ(reqs[0].odm_.size(), 2);
  ASSERT_EQ(reqs[1].pt_.size(), 1);
  ASSERT_EQ(reqs[1].odm_.size(), 1);

  for (auto const& j : reqs[0].pt_) {
    EXPECT_EQ(j.start_time_, n::unixtime_t{10h + 17min});
    EXPECT_EQ(j.dest_time_, n::unixtime_t{11h});
    ASSERT_EQ(j.legs_.size(), 1);
    ASSERT_TRUE(
        std::holds_alternative<n::routing::offset>(j.legs_.front().uses_));
    EXPECT_EQ(std::get<n::routing::offset>(j.legs_.front().uses_).duration_,
              30min);
    EXPECT_EQ(
        std::get<n::routing::offset>(j.legs_.front().uses_).transport_mode_id_,
        kWalk);
  }

  EXPECT_EQ(reqs[1].odm_[0].start_time_, n::unixtime_t{10h + 14min});
  EXPECT_EQ(reqs[1].odm_[0].dest_time_, n::unixtime_t{11h});
  EXPECT_EQ(reqs[1].odm_[0].transfers_, 2);
  ASSERT_EQ(reqs[1].odm_[0].legs_.size(), 2);
  ASSERT_TRUE(std::holds_alternative<n::routing::offset>(
      reqs[1].odm_[0].legs_[0].uses_));
  EXPECT_EQ(
      std::get<n::routing::offset>(reqs[1].odm_[0].legs_[0].uses_).duration_,
      6min);
  EXPECT_EQ(std::get<n::routing::offset>(reqs[1].odm_[0].legs_[0].uses_)
                .transport_mode_id_,
            kODM);
  ASSERT_TRUE(std::holds_alternative<n::routing::offset>(
      reqs[1].odm_[0].legs_[1].uses_));
  EXPECT_EQ(
      std::get<n::routing::offset>(reqs[1].odm_[0].legs_[1].uses_).duration_,
      5min);
  EXPECT_EQ(std::get<n::routing::offset>(reqs[1].odm_[0].legs_[1].uses_)
                .transport_mode_id_,
            kWalk);

  EXPECT_FALSE(reqs[0].odm_to_dom_.test(0));
  EXPECT_TRUE(reqs[0].odm_to_dom_.test(1));
  EXPECT_TRUE(reqs[1].odm_to_dom_.test(0));
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

constexpr auto const prima_state_init =
    R"({"start":{"lat":0E0,"lon":0E0},"target":{"lat":1E0,"lon":1E0},"startBusStops":[{"coordinates":{"lat":1E-1,"lon":1E-1},"times":["1970-01-01T11:00+0000","1970-01-01T12:00+0000"]},{"coordinates":{"lat":2E-1,"lon":2E-1},"times":["1970-01-01T12:00+0000"]}],"targetBusStops":[{"coordinates":{"lat":3.0000000000000004E-1,"lon":3.0000000000000004E-1},"times":["1970-01-01T13:00+0000"]},{"coordinates":{"lat":4E-1,"lon":4E-1},"times":["1970-01-01T14:00+0000"]}],"times":["1970-01-01T10:00+0000","1970-01-01T11:00+0000"],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

constexpr auto const blacklisting_response = R"(
{
  "startBusStops": [[true,false],[true]],
  "targetBusStops": [[true],[false]],
  "times": [false,true]
}
)";

constexpr auto const prima_state_blacklist =
    R"({"start":{"lat":0E0,"lon":0E0},"target":{"lat":1E0,"lon":1E0},"startBusStops":[{"coordinates":{"lat":1E-1,"lon":1E-1},"times":["1970-01-01T11:00+0000"]},{"coordinates":{"lat":2E-1,"lon":2E-1},"times":["1970-01-01T12:00+0000"]}],"targetBusStops":[{"coordinates":{"lat":3.0000000000000004E-1,"lon":3.0000000000000004E-1},"times":["1970-01-01T13:00+0000"]}],"times":["1970-01-01T11:00+0000"],"startFixed":true,"capacities":{"wheelchairs":1,"bikes":0,"passengers":1,"luggage":0}})";

constexpr auto const whitelisting_response = R"(
{
  "startBusStops": [["1970-01-01T10:45+0000"],["1970-01-01T12:00+0000"]],
  "targetBusStops": [["1970-01-01T13:05+0000"]],
  "times": ["1970-01-01T11:30+0000"]
}
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

  auto p = prima_state{
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

  EXPECT_EQ(prima_state_init, p.get_msg_str(tt));
  p.blacklist_update(blacklisting_response);
  EXPECT_EQ(prima_state_blacklist, p.get_msg_str(tt));

  std::cout << "before whitelisting" << std::endl;
  p.whitelist_update(whitelisting_response);

  std::cout << p.get_msg_str(tt) << "\n";
}
