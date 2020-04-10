#include "gtest/gtest.h"

#include "motis/routing/label/criteria/late_connections.h"

namespace motis::routing {

TEST(routing_late_connections_util, night_travel_duration) {
  ASSERT_EQ(0, night_travel_duration(0, 200, 100, 100));
  ASSERT_EQ(0, night_travel_duration(150, 150, 100, 200));

  auto f = [](uint16_t const b, uint16_t const e) {
    return night_travel_duration(b, e, 60, 360);
  };

  ASSERT_EQ(0, f(0, 59));
  ASSERT_EQ(0, f(0, 60));
  ASSERT_EQ(1, f(0, 61));
  ASSERT_EQ(0, f(60, 60));
  ASSERT_EQ(0, f(100, 100));
  ASSERT_EQ(0, f(360, 360));
  ASSERT_EQ(1, f(100, 101));
  ASSERT_EQ(300, f(60, 360));
  ASSERT_EQ(300, f(59, 361));
  ASSERT_EQ(1, f(359, 361));
  ASSERT_EQ(0, f(360, 361));
  ASSERT_EQ(0, f(361, 1500));
  ASSERT_EQ(1, f(361, 1501));

  ASSERT_EQ(300, f(0, 1441));
  ASSERT_EQ(600, f(0, 2881));
  ASSERT_EQ(601, f(0, 2941));
}

TEST(routing_late_connections_util, night_travel_duration_overnight) {
  auto f = [](uint16_t const b, uint16_t const e) {
    return night_travel_duration(b, e, 1380, 240);
  };

  ASSERT_EQ(240, f(0, 240));
  ASSERT_EQ(240, f(0, 241));
  ASSERT_EQ(0, f(240, 1380));
  ASSERT_EQ(0, f(241, 1379));
  ASSERT_EQ(1, f(240, 1381));
  ASSERT_EQ(1, f(1379, 1381));
  ASSERT_EQ(1, f(1380, 1381));
  ASSERT_EQ(1, f(1381, 1382));
  ASSERT_EQ(0, f(1380, 1380));
  ASSERT_EQ(0, f(1381, 1381));
  ASSERT_EQ(0, f(240, 240));
  ASSERT_EQ(120, f(1380, 1500));
  ASSERT_EQ(119, f(1381, 1500));
  ASSERT_EQ(300, f(1380, 1680));
  ASSERT_EQ(300, f(1379, 1681));

  ASSERT_EQ(301, f(0, 1441));
  ASSERT_EQ(601, f(0, 2881));
  ASSERT_EQ(840, f(0, 3121));
}

}  // namespace motis::routing
