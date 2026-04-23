#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis/endpoints/initial.h"

#include "../test_case.h"

TEST(initial, motis_version) {
  // Pick any test case, preferably the smallest
  auto [d, _config] = get_test_case<test_case::FFM_one_to_many>();
  auto const version = "v1.2.3-test";
  d.init_initial(version);  // Must be invoked by motis_instance()
  auto const ep = utl::init_from<motis::ep::initial>(d).value();

  auto const res = ep("");

  EXPECT_EQ(version, res.serverConfig_.motisVersion_);
}
