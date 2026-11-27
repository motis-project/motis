#include "gtest/gtest.h"

#include "nigiri/types.h"

#include "motis/qa.h"

using namespace date;
using namespace motis;
using namespace nigiri;

TEST(qa, same_journey_later) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 3_hours},
       .transfers_ = 0U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 1_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
       .transfers_ = 0U}};

  EXPECT_EQ(qa::rate(a, b, qa::kStartEndTransfer), 0.0);
}

TEST(qa, test0) {
  auto a = std::vector<api::Itinerary>{};
  auto const b = std::vector<api::Itinerary>{};

  EXPECT_DOUBLE_EQ(0.0, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(0.0, qa::rate(b, a, qa::kStartEndTransfer));

  a.push_back(
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U});

  EXPECT_DOUBLE_EQ(qa::kMinRating, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(qa::kMaxRating, qa::rate(b, a, qa::kStartEndTransfer));
}

TEST(qa, test1) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
       .transfers_ = 0U},
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
       .transfers_ = 1U}};

  EXPECT_DOUBLE_EQ(-0.342008418450396, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(0.342008418450396, qa::rate(b, a, qa::kStartEndTransfer));
}

TEST(qa, test2) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
       .transfers_ = 1U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
       .transfers_ = 0U},
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
       .transfers_ = 1U}};

  EXPECT_DOUBLE_EQ(-15.116357209650212, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(15.116357209650212, qa::rate(b, a, qa::kStartEndTransfer));
}

TEST(qa, test3) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
       .transfers_ = 1U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
       .transfers_ = 1U}};

  EXPECT_DOUBLE_EQ(-31.478651986610316, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(31.478651986610316, qa::rate(b, a, qa::kStartEndTransfer));
}

TEST(qa, test4) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 11_hours},
       .transfers_ = 1U},
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 10_hours + 45_minutes},
       .transfers_ = 3U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
       .transfers_ = 0U},
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
       .transfers_ = 1U}};

  EXPECT_DOUBLE_EQ(-20.839157331515052, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(20.839157331515052, qa::rate(b, a, qa::kStartEndTransfer));
}

TEST(qa, test5) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 1_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 23_hours},
       .transfers_ = 0U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 22_hours},
       .transfers_ = 1U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 3_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 21_hours},
       .transfers_ = 2U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 4_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 20_hours},
       .transfers_ = 3U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 5_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 19_hours},
       .transfers_ = 4U},
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 6_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 18_hours},
       .transfers_ = 5U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 33_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 11_hours + 34_minutes},
       .transfers_ = 0U},
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 9_hours + 45_minutes},
       .endTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 12_hours + 16_minutes},
       .transfers_ = 1U}};

  EXPECT_DOUBLE_EQ(32.37407751772509, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(-32.37407751772509, qa::rate(b, a, qa::kStartEndTransfer));
}

TEST(qa, walking_time) {
  auto const i = api::Itinerary{
      .legs_ = {{.mode_ = api::ModeEnum::WALK, .duration_ = 111},
                {.mode_ = api::ModeEnum::TRANSIT, .duration_ = 222},
                {.mode_ = api::ModeEnum::WALK, .duration_ = 333},
                {.mode_ = api::ModeEnum::TRANSIT, .duration_ = 444},
                {.mode_ = api::ModeEnum::WALK, .duration_ = 555}}};
  EXPECT_EQ(qa::criterion::kDefaultWalkingTime(i), 17.0);  // 999s -> ~17min
}

TEST(qa, different_criteria) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 3_hours},
       .transfers_ = 0U,
       .legs_ = {{.mode_ = api::ModeEnum::WALK, .duration_ = 300},
                 {.mode_ = api::ModeEnum::TRANSIT, .duration_ = 3000},
                 {.mode_ = api::ModeEnum::WALK, .duration_ = 300}}}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 1_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 2_hours},
       .transfers_ = 0U,
       .legs_ = {{.mode_ = api::ModeEnum::WALK, .duration_ = 600},
                 {.mode_ = api::ModeEnum::TRANSIT, .duration_ = 2400},
                 {.mode_ = api::ModeEnum::WALK, .duration_ = 600}}}};

  EXPECT_EQ(qa::rate(a, b, qa::kStartEndTransfer), 0.0);
  EXPECT_DOUBLE_EQ(-0.12416657898367944,
                   qa::rate(a, b, qa::kStartEndTransferWalk));
}

TEST(qa, minor_improvement) {
  auto const a = std::vector<api::Itinerary>{
      {.startTime_ = unixtime_t{sys_days{2024_y / June / 10} + 10_hours},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 12_hours},
       .transfers_ = 0U}};

  auto const b = std::vector<api::Itinerary>{
      {.startTime_ =
           unixtime_t{sys_days{2024_y / June / 10} + 10_hours + 1_minutes},
       .endTime_ = unixtime_t{sys_days{2024_y / June / 10} + 15_hours},
       .transfers_ = 5U}};

  EXPECT_DOUBLE_EQ(-49.460622942808466, qa::rate(a, b, qa::kStartEndTransfer));
  EXPECT_DOUBLE_EQ(49.460622942808466, qa::rate(b, a, qa::kStartEndTransfer));
}