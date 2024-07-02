#include "gtest/gtest.h"

#include "icc/parse_location.h"

TEST(icc, parse_location_with_level) {
  auto const parsed = icc::parse_location("-123.1,44.2,-1.5");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ((osr::location{{-123.1, 44.2}, osr::to_level(-1.5F)}), *parsed);
}

TEST(icc, parse_location_no_level) {
  auto const parsed = icc::parse_location("-23.1,45.2");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ((osr::location{{-23.1, 45.2}, osr::to_level(0.F)}), *parsed);
}