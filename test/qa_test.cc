#include "gtest/gtest.h"

#include "motis/qa/qa.h"

TEST(qa, test0) {
  auto a = pareto_set<journey>{};
  auto b = pareto_set<journey>{};

  EXPECT_DOUBLE_EQ(0.0, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(0.0, qa::rate(b, a));

  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
         .transfers_ = 0U});

  EXPECT_DOUBLE_EQ(qa::kMaxRating, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(qa::kMinRating, qa::rate(b, a));
}

TEST(qa, test1) {
  auto a = pareto_set<journey>{};
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
         .transfers_ = 0U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
         .transfers_ = 0U});

  auto b = pareto_set<journey>{};
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
         .transfers_ = 0U});
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
         .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(0.342008418450396, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-0.342008418450396, qa::rate(b, a));
}

TEST(qa, test2) {
  auto a = pareto_set<journey>{};
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
         .transfers_ = 0U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
         .transfers_ = 1U});

  auto b = pareto_set<journey>{};
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
         .transfers_ = 0U});
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
         .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(15.116357209650212, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-15.116357209650212, qa::rate(b, a));
}

TEST(qa, test3) {
  auto a = pareto_set<journey>{};
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
         .transfers_ = 0U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
         .transfers_ = 1U});

  auto b = pareto_set<journey>{};
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
         .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(31.478651986610316, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-31.478651986610316, qa::rate(b, a));
}

TEST(qa, test4) {
  auto a = pareto_set<journey>{};
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
         .transfers_ = 0U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
         .transfers_ = 1U});
  a.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 10_hours + 45_minutes},
         .transfers_ = 3U});

  auto b = pareto_set<journey>{};
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
         .transfers_ = 0U});
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
         .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(20.839157331515052, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(-20.839157331515052, qa::rate(b, a));
}

TEST(qa, test5) {
  auto a = pareto_set<journey>{};
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 1_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 23_hours},
         .transfers_ = 0U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 22_hours},
         .transfers_ = 1U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 3_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 21_hours},
         .transfers_ = 2U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 4_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 20_hours},
         .transfers_ = 3U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 5_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 19_hours},
         .transfers_ = 4U});
  a.add({.start_time_ = unixtime_t{sys_days{2024_y / June / 10} + 6_hours},
         .dest_time_ = unixtime_t{sys_days{2024_y / June / 10} + 18_hours},
         .transfers_ = 5U});

  auto b = pareto_set<journey>{};
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
         .transfers_ = 0U});
  b.add({.start_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
         .dest_time_ =
             unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
         .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(-32.37407751772509, qa::rate(a, b));
  EXPECT_DOUBLE_EQ(32.37407751772509, qa::rate(b, a));
}