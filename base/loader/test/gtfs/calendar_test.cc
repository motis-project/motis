#include "gtest/gtest.h"

#include "motis/loader/gtfs/calendar.h"
#include "motis/loader/gtfs/files.h"

#include "./resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;
using namespace date;

TEST(loader_gtfs_calendar, read_calendar_example_data) {
  auto calendar =
      read_calendar(loaded_file{SCHEDULES / "example" / CALENDAR_FILE});

  EXPECT_EQ(2, calendar.size());

  EXPECT_TRUE(calendar["WE"].week_days_.test(0));
  EXPECT_FALSE(calendar["WE"].week_days_.test(1));
  EXPECT_FALSE(calendar["WE"].week_days_.test(2));
  EXPECT_FALSE(calendar["WE"].week_days_.test(3));
  EXPECT_FALSE(calendar["WE"].week_days_.test(4));
  EXPECT_FALSE(calendar["WE"].week_days_.test(5));
  EXPECT_TRUE(calendar["WE"].week_days_.test(6));

  EXPECT_FALSE(calendar["WD"].week_days_.test(0));
  EXPECT_TRUE(calendar["WD"].week_days_.test(1));
  EXPECT_TRUE(calendar["WD"].week_days_.test(2));
  EXPECT_TRUE(calendar["WD"].week_days_.test(3));
  EXPECT_TRUE(calendar["WD"].week_days_.test(4));
  EXPECT_TRUE(calendar["WD"].week_days_.test(5));
  EXPECT_FALSE(calendar["WD"].week_days_.test(6));

  EXPECT_TRUE(2006_y / 7 / 1 ==
              date::year_month_day{calendar["WE"].first_day_});
  EXPECT_TRUE(2006_y / 7 / 31 ==
              date::year_month_day{calendar["WE"].last_day_});
  EXPECT_TRUE(2006_y / 7 / 1 ==
              date::year_month_day{calendar["WD"].first_day_});
  EXPECT_TRUE(2006_y / 7 / 31 ==
              date::year_month_day{calendar["WD"].last_day_});
}
