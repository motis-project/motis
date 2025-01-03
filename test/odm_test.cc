#include "gtest/gtest.h"

#include <algorithm>

#include "utl/zip.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/special_stations.h"

#include "motis/odm/mix.h"

namespace n = nigiri;
using namespace motis::odm;
using namespace std::chrono_literals;

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
  auto oev = n::routing::journey{
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
  pt_journeys.add(n::routing::journey{oev});

  auto oev_taxi = n::routing::journey{
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
      oev_taxi,
      direct_taxi(n::unixtime_t{10h + 17min}, n::unixtime_t{10h + 27min}),
      direct_taxi(n::unixtime_t{10h + 43min}, n::unixtime_t{10h + 53min}),
      direct_taxi(n::unixtime_t{10h + 50min}, n::unixtime_t{11h + 00min}),
      direct_taxi(n::unixtime_t{10h + 00min}, n::unixtime_t{10h + 10min}),
      direct_taxi(n::unixtime_t{11h + 00min}, n::unixtime_t{10h + 10min})};

  mix(pt_journeys, odm_journeys);

  ASSERT_EQ(odm_journeys.size(), 2U);
  EXPECT_NE(std::find_if(begin(odm_journeys), end(odm_journeys),
                         [&](auto const& j) { return equal(j, oev); }),
            end(odm_journeys));
  EXPECT_NE(std::find_if(begin(odm_journeys), end(odm_journeys),
                         [&](auto const& j) { return equal(j, oev_taxi); }),
            end(odm_journeys));
}