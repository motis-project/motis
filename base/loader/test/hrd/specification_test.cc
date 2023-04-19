#include <cinttypes>
#include <cstring>

#include "gtest/gtest.h"

#include "utl/parser/cstr.h"
#include "utl/verify.h"

#include "motis/loader/parser_error.h"

#include "./paths.h"
#include "./test_spec_test.h"

using namespace utl;

namespace motis::loader::hrd {

TEST(loader_hrd_specification, parse_specification) {
  const auto fahrten = SCHEDULES / "hand-crafted" / "fahrten";
  test_spec const services_file(fahrten, "services-1.101");

  const auto specs = services_file.get_specs();
  ASSERT_TRUE(specs.size() == 1);

  auto& spec = specs[0];
  ASSERT_TRUE(!spec.is_empty());
  ASSERT_TRUE(spec.valid());
  ASSERT_TRUE(!spec.internal_service_.empty());
  ASSERT_TRUE(spec.traffic_days_.size() == 1);
  ASSERT_TRUE(spec.categories_.size() == 1);
  ASSERT_TRUE(spec.attributes_.size() == 3);
  ASSERT_TRUE(spec.stops_.size() == 6);
}

TEST(loader_hrd_specification, parse_hrd_service_invalid_traffic_days) {
  bool catched = false;
  try {
    test_spec(SCHEDULES / "hand-crafted" / "fahrten", "services-3.101")
        .get_hrd_services(hrd_5_00_8);
    test_spec(SCHEDULES / "hand-crafted_new" / "fahrten", "services-3.txt")
        .get_hrd_services(hrd_5_20_26);
  } catch (std::runtime_error const& e) {
    catched = true;
  }
  ASSERT_TRUE(catched);
}

}  // namespace motis::loader::hrd
