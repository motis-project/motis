#include "gtest/gtest.h"

#include "motis/loader/gtfs/calendar_date.h"
#include "motis/loader/gtfs/files.h"

#include "resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;
using namespace date;

TEST(loader_gtfs_calendar_date, read_calendar_date_example_data) {
  auto dates = read_calendar_date(
      loaded_file{SCHEDULES / "example" / CALENDAR_DATES_FILE});

  EXPECT_EQ(2, dates.size());
  EXPECT_EQ(2, dates["WD"].size());
  EXPECT_EQ(2, dates["WE"].size());

  EXPECT_EQ(2006_y, date::year_month_day{dates["WD"][0].day_}.year());
  EXPECT_EQ(July, date::year_month_day{dates["WD"][0].day_}.month());
  EXPECT_EQ(date::day{3}, date::year_month_day{dates["WD"][0].day_}.day());
  EXPECT_EQ(calendar_date::REMOVE, dates["WD"][0].type_);

  EXPECT_EQ(2006_y, date::year_month_day{dates["WE"][0].day_}.year());
  EXPECT_EQ(July, date::year_month_day{dates["WE"][0].day_}.month());
  EXPECT_EQ(date::day{3}, date::year_month_day{dates["WE"][0].day_}.day());
  EXPECT_EQ(calendar_date::ADD, dates["WE"][0].type_);

  EXPECT_EQ(2006_y, date::year_month_day{dates["WD"][1].day_}.year());
  EXPECT_EQ(July, date::year_month_day{dates["WD"][1].day_}.month());
  EXPECT_EQ(date::day{4}, date::year_month_day{dates["WD"][1].day_}.day());
  EXPECT_EQ(calendar_date::REMOVE, dates["WD"][1].type_);

  EXPECT_EQ(2006_y, date::year_month_day{dates["WE"][1].day_}.year());
  EXPECT_EQ(July, date::year_month_day{dates["WE"][1].day_}.month());
  EXPECT_EQ(date::day{4}, date::year_month_day{dates["WE"][1].day_}.day());
  EXPECT_EQ(calendar_date::ADD, dates["WE"][1].type_);
}
