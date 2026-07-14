#include "gtest/gtest.h"

#include "nigiri/types.h"

#include "motis/qa.h"

#include "motis/timetable/time_conv.h"

using namespace date;
using namespace motis;
using namespace nigiri;

auto const criteria = std::vector<qa::criterion_t>{
    qa::criterion::start_time<1.0>, qa::criterion::end_time<1.0>,
    qa::criterion::transfers<30.0>};

TEST(qa, test0) {
  auto a = std::vector<api::Itinerary>{};
  auto b = std::vector<api::Itinerary>{};

  EXPECT_DOUBLE_EQ(0.0, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(0.0, qa::rate(b, a, criteria));

  a.push_back(api::Itinerary{
      .startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
      .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
      .transfers_ = 0U});

  EXPECT_DOUBLE_EQ(qa::kMaxRating, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(qa::kMinRating, qa::rate(b, a, criteria));
}

TEST(qa, test1) {
  auto a = std::vector<api::Itinerary>{};
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U});
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U});

  auto b = std::vector<api::Itinerary>{};
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        33_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours +
                                      34_minutes},
               .transfers_ = 0U});
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        45_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours +
                                      16_minutes},
               .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(0.342008418450396, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(-0.342008418450396, qa::rate(b, a, criteria));
}

TEST(qa, test2) {
  auto a = std::vector<api::Itinerary>{};
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U});
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
       .transfers_ = 1U});

  auto b = std::vector<api::Itinerary>{};
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        33_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours +
                                      34_minutes},
               .transfers_ = 0U});
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        45_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours +
                                      16_minutes},
               .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(15.116357209650212, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(-15.116357209650212, qa::rate(b, a, criteria));
}

TEST(qa, test3) {
  auto a = std::vector<api::Itinerary>{};
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U});
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
       .transfers_ = 1U});

  auto b = std::vector<api::Itinerary>{};
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        45_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours +
                                      16_minutes},
               .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(31.478651986610316, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(-31.478651986610316, qa::rate(b, a, criteria));
}

TEST(qa, test4) {
  auto a = std::vector<api::Itinerary>{};
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U});
  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
       .transfers_ = 1U});
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        45_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours +
                                      45_minutes},
               .transfers_ = 3U});

  auto b = std::vector<api::Itinerary>{};
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        33_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours +
                                      34_minutes},
               .transfers_ = 0U});
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        45_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours +
                                      16_minutes},
               .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(20.839157331515052, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(-20.839157331515052, qa::rate(b, a, criteria));
}

TEST(qa, test5) {
  auto a = std::vector<api::Itinerary>{};
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 1_hours},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 23_hours},
               .transfers_ = 0U});
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 22_hours},
               .transfers_ = 1U});
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 3_hours},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 21_hours},
               .transfers_ = 2U});
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 4_hours},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 20_hours},
               .transfers_ = 3U});
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 5_hours},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 19_hours},
               .transfers_ = 4U});
  a.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 6_hours},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 18_hours},
               .transfers_ = 5U});

  auto b = std::vector<api::Itinerary>{};
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        33_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours +
                                      34_minutes},
               .transfers_ = 0U});
  b.push_back({.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 9_hours +
                                        45_minutes},
               .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours +
                                      16_minutes},
               .transfers_ = 1U});

  EXPECT_DOUBLE_EQ(-32.37407751772509, qa::rate(a, b, criteria));
  EXPECT_DOUBLE_EQ(32.37407751772509, qa::rate(b, a, criteria));
}