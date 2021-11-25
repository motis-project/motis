#include "gtest/gtest.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/timezone.h"

#include "motis/loader/timezone_util.h"

namespace motis::loader {

// schedule begin time := 1440
// schedule interval := {}
// (unmodified) season interval := ()
// (final) season interval := []
// MAD := MINUTES_A_DAY == 1440
// invalid time := INV

TEST(core_timezone, gmt_plus_one) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 0;
  auto const day_idx_schedule_last_day = 6;

  auto const day_idx_season_first_day = 2;
  auto const day_idx_season_last_day = 4;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  ASSERT_EQ(general_offset, tz.general_offset_);
  ASSERT_EQ(season_offset, tz.season_.offset_);

  // { MAD 5*MAD [ 6*MAD 7*MAD 8*MAD ] 9*MAD 10*MAD }
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 2,
                 minutes_after_midnight_season_begin - general_offset),
            tz.season_.begin_);
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 4,
                 minutes_after_midnight_season_end - season_offset),
            tz.season_.end_);

  ASSERT_EQ(SCHEDULE_OFFSET_MINUTES, tz.to_motis_time(0, 60).ts());
  ASSERT_EQ(SCHEDULE_OFFSET_MINUTES + 4 * MINUTES_A_DAY + 180,
            tz.to_motis_time(4, 240).ts());
}

TEST(core_timezone, gmt_minus_one) {
  auto const general_offset = -60;
  auto const season_offset = -120;

  auto const day_idx_schedule_first_day = 0;
  auto const day_idx_schedule_last_day = 6;

  auto const day_idx_season_first_day = 2;
  auto const day_idx_season_last_day = 4;

  auto const minutes_after_midnight_season_begin = 0;
  auto const minutes_after_midnight_season_end = 60;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  ASSERT_EQ(general_offset, tz.general_offset_);
  ASSERT_EQ(season_offset, tz.season_.offset_);

  // { MAD 5*MAD [ 6*MAD 7*MAD 8*MAD ] 9*MAD 10*MAD }
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 2,
                 minutes_after_midnight_season_begin - general_offset),
            tz.season_.begin_);
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 4,
                 minutes_after_midnight_season_end - season_offset),
            tz.season_.end_);

  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS, 120), tz.to_motis_time(0, 60));
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 4, 180), tz.to_motis_time(4, 120));
}

TEST(core_timezone, season_begin_end_overlaps_schedule_period) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 1;
  auto const day_idx_schedule_last_day = 5;

  auto const day_idx_season_first_day = 0;
  auto const day_idx_season_last_day = 6;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  // from: [ INV {  5*MAD  6*MAD 7*MAD 8*MAD 9*MAD }  INV ]
  // to:     INV [{ 5*MAD  6*MAD 7*MAD 8*MAD 9*MAD }] INV
  ASSERT_EQ(0, tz.season_.begin_.ts());
  ASSERT_EQ(MAX_TIME - season_offset, tz.season_.end_);
}

TEST(core_timezone, season_end_before_schedule_period) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 3;
  auto const day_idx_schedule_last_day = 6;

  auto const day_idx_season_first_day = 2;
  auto const day_idx_season_last_day = 1;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  //  [ INV INV INV ] { 5*MAD  6*MAD 7*MAD 8*MAD }
  ASSERT_EQ(INVALID_TIME, tz.season_.begin_);
  ASSERT_EQ(INVALID_TIME, tz.season_.end_);
  ASSERT_EQ(season::INVALID_OFFSET, tz.season_.offset_);
}

TEST(core_timezone, season_begin_after_schedule_period) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 2;
  auto const day_idx_schedule_last_day = 6;

  auto const day_idx_season_first_day = 0;
  auto const day_idx_season_last_day = 1;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  //  { 5*MAD  6*MAD 7*MAD 8*MAD } [ INV INV INV ]
  ASSERT_EQ(INVALID_TIME, tz.season_.begin_);
  ASSERT_EQ(INVALID_TIME, tz.season_.end_);
  ASSERT_EQ(season::INVALID_OFFSET, tz.season_.offset_);
}

TEST(core_timezone, move_season_begin_to_schedule_period_begin) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 1;
  auto const day_idx_schedule_last_day = 4;

  auto const day_idx_season_first_day = 0;
  auto const day_idx_season_last_day = 2;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  // from: [ INV {  5*MAD  6*MAD ] 7*MAD 8*MAD } INV INV
  // to:     INV [{ 5*MAD  6*MAD ] 7*MAD 8*MAD } INV INV
  ASSERT_EQ(0, tz.season_.begin_.ts());
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 1,
                 minutes_after_midnight_season_end - season_offset),
            tz.season_.end_);
}

TEST(core_timezone, move_season_end_to_schedule_period_end) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 1;
  auto const day_idx_schedule_last_day = 4;

  auto const day_idx_season_first_day = 3;
  auto const day_idx_season_last_day = 5;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  // from:   INV { 5*MAD  6*MAD [ 7*MAD 8*MAD }  INV ] INV
  // to:     INV { 5*MAD  6*MAD [ 7*MAD 8*MAD }] INV   INV
  ASSERT_EQ(time(SCHEDULE_OFFSET_DAYS + 2,
                 minutes_after_midnight_season_begin - general_offset),
            tz.season_.begin_);
  ASSERT_EQ(MAX_TIME - season_offset, tz.season_.end_);
}

bool is_invalid_time(int day_idx, int minutes_after_midnight,
                     timezone const& tz) {
  return tz.to_motis_time(day_idx, minutes_after_midnight) == INVALID_TIME;
}

TEST(core_timezone, invalid_event) {
  auto const general_offset = 60;
  auto const season_offset = 120;

  auto const day_idx_schedule_first_day = 0;
  auto const day_idx_schedule_last_day = 6;

  auto const day_idx_season_first_day = 1;
  auto const day_idx_season_last_day = 5;

  auto const minutes_after_midnight_season_begin = 120;
  auto const minutes_after_midnight_season_end = 180;

  timezone const tz = create_timezone(
      general_offset, season_offset, day_idx_schedule_first_day,
      day_idx_schedule_last_day, day_idx_season_first_day,
      day_idx_season_last_day, minutes_after_midnight_season_begin,
      minutes_after_midnight_season_end);

  // { MAD [ 5*MAD  6*MAD 7*MAD 8*MAD  9*MAD ] 10*MAD }
  EXPECT_FALSE(is_invalid_time(1, 119, tz));
  EXPECT_FALSE(is_invalid_time(1, 180, tz));
  EXPECT_FALSE(is_invalid_time(1, 181, tz));
  EXPECT_TRUE(is_invalid_time(1, 120, tz));
  EXPECT_TRUE(is_invalid_time(1, 179, tz));
}

}  // namespace motis::loader
