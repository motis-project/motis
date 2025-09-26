#include "gtest/gtest.h"

#include "motis/odm/mixer.h"
#include "motis/odm/odm.h"

using namespace std::string_view_literals;
using namespace motis::odm;

TEST(odm, tally) {
  auto const ct = std::vector<cost_threshold>{{0, 30}, {1, 1}, {10, 2}};
  EXPECT_EQ(0, tally(0, ct));
  EXPECT_EQ(30, tally(1, ct));
  EXPECT_EQ(43, tally(12, ct));
}