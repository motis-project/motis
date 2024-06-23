#include "gtest/gtest.h"

#include "icc/match_platforms.h"

TEST(icc, get_track) {
  ASSERT_FALSE(icc::get_track("a:").has_value());

  auto const track = icc::get_track("a:232");
  ASSERT_TRUE(track.has_value());
  EXPECT_EQ("232", *track);

  auto const track_1 = icc::get_track("232");
  ASSERT_TRUE(track_1.has_value());
  EXPECT_EQ("232", *track_1);
}