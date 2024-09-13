#include "gtest/gtest.h"

#include "motis/match_platforms.h"

TEST(motis, get_track) {
  ASSERT_FALSE(motis::get_track("a:").has_value());

  auto const track = motis::get_track("a:232");
  ASSERT_TRUE(track.has_value());
  EXPECT_EQ("232", *track);

  auto const track_1 = motis::get_track("232");
  ASSERT_TRUE(track_1.has_value());
  EXPECT_EQ("232", *track_1);
}