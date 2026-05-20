#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis/endpoints/initial.h"

#include "../test_case.h"

TEST(initial, server_config) {
  // Pick any test case, preferably the smallest
  auto [d, _config] = get_test_case<test_case::FFM_one_to_many>();
  auto const version = "v1.2.3-test";
  d.init_initial(version);  // Must be invoked by motis_instance()
  auto const ep = utl::init_from<motis::ep::initial>(d).value();

  auto const res = ep("");

  auto const& config = res.serverConfig_;
  EXPECT_EQ(version, config.motisVersion_);
  EXPECT_EQ(128.0, config.maxOneToManySize_);
  EXPECT_EQ(90.0, config.maxOneToAllTravelTimeLimit_);
  EXPECT_EQ(3600.0, config.maxPrePostTransitTimeLimit_);
  EXPECT_EQ(21600.0, config.maxDirectTimeLimit_);
  EXPECT_TRUE(config.hasRoutedTransfers_);
  EXPECT_TRUE(config.hasStreetRouting_);
  EXPECT_FALSE(config.hasElevation_);
  EXPECT_FALSE(config.shapesDebugEnabled_);
}
