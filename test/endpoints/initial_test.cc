#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis/endpoints/initial.h"
#include "motis/import.h"

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

TEST(initial, osm_only_fallback) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_osm_only_initial", ec);

  auto const c = config{.osm_ = {"test/resources/test_case.osm.pbf"},
                        .street_routing_ = true};
  import(c, "test/data_osm_only_initial");

  auto d = data{"test/data_osm_only_initial", c};
  d.init_initial("v1.2.3-test");
  auto const ep = utl::init_from<motis::ep::initial>(d).value();

  auto const res = ep("");

  EXPECT_NEAR(49.9943077, res.lat_, 1e-6);
  EXPECT_NEAR(8.6573986, res.lon_, 1e-6);
  EXPECT_EQ(10.0, res.zoom_);
  EXPECT_TRUE(res.serverConfig_.hasStreetRouting_);
}
