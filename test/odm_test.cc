#include "gtest/gtest.h"

#include "utl/zip.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/special_stations.h"

#include "motis/odm/calibration/json.h"
#include "motis/odm/mix.h"
#include "motis/odm/odm.h"

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

bool equal(n::routing::journey const& a, n::routing::journey const& b) {
  if (std::tie(a.start_time_, a.dest_time_, a.dest_, a.transfers_) !=
          std::tie(b.start_time_, b.dest_time_, b.dest_, b.transfers_) ||
      a.legs_.size() != b.legs_.size()) {
    return false;
  }

  auto const equal_leg = [](auto const& l1, auto& l2) {
    return std::tie(l1.from_, l1.to_, l1.dep_time_, l1.arr_time_) ==
               std::tie(l2.from_, l2.to_, l2.dep_time_, l2.arr_time_) &&
           l1.uses_.index() == l2.uses_.index();
  };

  for (auto const [x, y] : utl::zip(a.legs_, b.legs_)) {
    if (!equal_leg(x, y)) {
      return false;
    }
  }

  return true;
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
                         [&](auto const& j) { return equal(j, pt); }),
            end(odm_journeys));
  EXPECT_NE(std::find_if(begin(odm_journeys), end(odm_journeys),
                         [&](auto const& j) { return equal(j, pt_taxi); }),
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
                         [&](auto const& j) { return equal(j, pt); }),
            end(odm_journeys));
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
}
