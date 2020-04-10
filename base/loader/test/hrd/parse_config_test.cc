#include "gtest/gtest.h"

#include "motis/loader/hrd/hrd_parser.h"

#include "./paths.h"

namespace motis::loader::hrd {

TEST(loader_hrd_config, config_detection_test) {
  hrd_parser p;
  auto const root_old_complete = SCHEDULES / "hand-crafted";
  auto const root_new_complete = SCHEDULES / "hand-crafted_new";

  ASSERT_TRUE(p.applicable(root_old_complete, hrd_5_00_8));
  ASSERT_FALSE(p.applicable(root_old_complete, hrd_5_20_26));
  ASSERT_TRUE(p.applicable(root_old_complete));

  ASSERT_TRUE(p.applicable(root_new_complete, hrd_5_20_26));
  ASSERT_FALSE(p.applicable(root_new_complete, hrd_5_00_8));
  ASSERT_TRUE(p.applicable(root_new_complete));

  auto const root_old_incomplete =
      SCHEDULES / "incomplete_schedules" / "incomplete_old";
  auto const root_new_incomplete =
      SCHEDULES / "incomplete_schedules" / "incomplete_new";
  auto const root_both_incomplete =
      SCHEDULES / "incomplete_schedules" / "incomplete_both";

  ASSERT_FALSE(p.applicable(root_old_incomplete));
  ASSERT_FALSE(p.applicable(root_new_incomplete));
  ASSERT_FALSE(p.applicable(root_both_incomplete));

  auto const missing_old = p.missing_files(root_old_incomplete);
  auto const missing_new = p.missing_files(root_new_incomplete);
  auto const missing_both = p.missing_files(root_both_incomplete);

  EXPECT_EQ(16, missing_old.size());
  EXPECT_EQ(14, missing_new.size());
  EXPECT_EQ("eckdaten.*", missing_both[0]);
}

}  // namespace motis::loader::hrd
