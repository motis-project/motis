#include <cinttypes>

#include "gtest/gtest.h"

#include "motis/schedule-format/RuleService_generated.h"

#include "motis/loader/hrd/parser/bitfields_parser.h"
#include "motis/loader/hrd/parser/through_services_parser.h"
#include "motis/loader/util.h"

#include "./paths.h"
#include "./test_spec_test.h"

namespace motis::loader::hrd {

TEST(loader_hrd_ts, multiple_rules) {
  test_spec b_spec(SCHEDULES / "ts-mss-hrd" / "stamm", "bitfield.101");
  auto hrd_bitfields = parse_bitfields(b_spec.lf_, hrd_5_00_8);
  test_spec ts_spec(SCHEDULES / "ts-mss-hrd" / "stamm", "durchbi.101");

  test_spec b_spec_new(SCHEDULES / "ts-mss-hrd_new" / "stamm", "bitfield.txt");
  auto hrd_bitfields_new = parse_bitfields(b_spec_new.lf_, hrd_5_20_26);
  test_spec ts_spec_new(SCHEDULES / "ts-mss-hrd_new" / "stamm", "durchbi.txt");

  service_rules rs_old;
  parse_through_service_rules(ts_spec.lf_, hrd_bitfields, rs_old, hrd_5_00_8);
  service_rules rs_new;
  parse_through_service_rules(ts_spec_new.lf_, hrd_bitfields_new, rs_new,
                              hrd_5_20_26);

  for (auto rs : {rs_old, rs_new}) {
    ASSERT_EQ(4, rs.size());
    auto it_1 = rs.find(std::make_pair(3040, raw_to_int<uint64_t>("07____")));
    ASSERT_TRUE(it_1 != end(rs));

    ASSERT_EQ(2, it_1->second.size());
    auto rule_info = it_1->second[0]->rule_info();
    ASSERT_EQ(8000267, rule_info.eva_num_1_);
    ASSERT_EQ(8000267, rule_info.eva_num_2_);
    auto it_b = hrd_bitfields.find(27351);
    ASSERT_TRUE(it_b != end(hrd_bitfields));
    ASSERT_EQ(it_b->second, rule_info.traffic_days_);
    ASSERT_EQ(RuleType_THROUGH, rule_info.type_);

    rule_info = it_1->second[1]->rule_info();
    ASSERT_EQ(8000028, rule_info.eva_num_1_);
    ASSERT_EQ(8000028, rule_info.eva_num_2_);
    it_b = hrd_bitfields.find(0);
    ASSERT_TRUE(it_b != end(hrd_bitfields));
    ASSERT_EQ(it_b->second, rule_info.traffic_days_);
    ASSERT_EQ(RuleType_THROUGH, rule_info.type_);

    auto it_2 = rs.find(std::make_pair(3056, raw_to_int<uint64_t>("07____")));
    ASSERT_TRUE(it_2 != end(rs));
    ASSERT_EQ(2, it_2->second.size());
    ASSERT_EQ(it_1->second[0].get(), it_2->second[0].get());
  }
}

}  // namespace motis::loader::hrd
