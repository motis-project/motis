#include "gtest/gtest.h"

#include <algorithm>
#include <filesystem>

#include "utl/init_from.h"

#include "motis/endpoints/adr/geocode.h"

#include "../test_case.h"

using namespace motis;

TEST(motis, stop_group_geocoding) {
  auto [d, _] = get_test_case<test_case::generated_stop_group_geocoding>();

  auto const geocode = utl::init_from<ep::geocode>(d).value();

  auto const g1 = geocode("/api/v1/geocode?text=Group%201");
  auto const g1_it =
      utl::find_if(g1, [](auto const& m) { return m.id_ == "test_G1"; });
  ASSERT_NE(end(g1), g1_it);
  ASSERT_TRUE(g1_it->modes_.has_value());
  EXPECT_NE(end(*g1_it->modes_), utl::find(*g1_it->modes_, api::ModeEnum::BUS));
  ASSERT_TRUE(g1_it->importance_.has_value());
  EXPECT_GT(*g1_it->importance_, 0.0);

  auto const g2 = geocode("/api/v1/geocode?text=Group%202");
  auto const g2_it =
      utl::find_if(g2, [](auto const& m) { return m.id_ == "test_G2"; });
  ASSERT_NE(end(g2), g2_it);
  ASSERT_TRUE(g2_it->modes_.has_value());
  EXPECT_NE(end(*g2_it->modes_),
            utl::find(*g2_it->modes_, api::ModeEnum::TRAM));
  ASSERT_TRUE(g2_it->importance_.has_value());
  EXPECT_GT(*g2_it->importance_, 0.0);
}
