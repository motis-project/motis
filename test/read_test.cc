#include "gtest/gtest.h"

#include "icc/parse_location.h"

TEST(icc, parse_location) {
  auto const parsed = icc::parse_location("-123.1,44.2,-1.5");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ((osr::location{{-123.1, 44.2}, osr::to_level(-1.5F)}), *parsed);
}