#include "gtest/gtest.h"

#include "motis/osr/max_distance.h"

using namespace motis;

TEST(motis, osr_max_distance_bike_sharing) {
  EXPECT_EQ(40.0, get_max_distance(osr::search_profile::kBikeSharing,
                                   osr_parameters{.cycling_speed_ = 8.0F},
                                   std::chrono::seconds{5}));
}
