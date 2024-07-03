#include "gtest/gtest.h"

#include "icc/parse_location.h"

using namespace icc;
using namespace date;

using namespace std::chrono_literals;

TEST(icc, parse_location_with_level) {
  auto const parsed = parse_location("-123.1,44.2,-1.5");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ((osr::location{{-123.1, 44.2}, osr::to_level(-1.5F)}), *parsed);
}

TEST(icc, parse_location_no_level) {
  auto const parsed = parse_location("-23.1,45.2");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ((osr::location{{-23.1, 45.2}, osr::to_level(0.F)}), *parsed);
}

TEST(icc, parse_date_time) {
  auto const t = get_date_time("06-28-2024", "7:06 PM");
  EXPECT_EQ(sys_days{2024_y / June / 28} + 19h + 6min, t);
}