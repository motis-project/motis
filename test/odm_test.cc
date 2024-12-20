#include "gtest/gtest.h"

#include "motis/odm/mix.h"

using namespace motis::odm;

TEST(motis, tally) {
  auto const ct = std::vector<cost_threshold>{{0, 30}, {1, 1}, {10, 2}};
  EXPECT_EQ(0, tally(0, ct));
  EXPECT_EQ(30, tally(1, ct));
  EXPECT_EQ(43, tally(12, ct));
}