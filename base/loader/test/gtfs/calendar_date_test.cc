#include "gtest/gtest.h"

#include "motis/loader/gtfs/calendar_date.h"
#include "motis/loader/gtfs/files.h"

#include "resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;

TEST(loader_gtfs_calendar_date, read_calendar_date_example_data) {
  auto dates = read_calendar_date(
      loaded_file{SCHEDULES / "example" / CALENDAR_DATES_FILE});

  EXPECT_EQ(2, dates.size());
  EXPECT_EQ(2, dates["WD"].size());
  EXPECT_EQ(2, dates["WE"].size());

  EXPECT_EQ(2006, dates["WD"][0].day_.year());
  EXPECT_EQ(7, dates["WD"][0].day_.month());
  EXPECT_EQ(3, dates["WD"][0].day_.day());
  EXPECT_EQ(calendar_date::REMOVE, dates["WD"][0].type_);

  EXPECT_EQ(2006, dates["WE"][0].day_.year());
  EXPECT_EQ(7, dates["WE"][0].day_.month());
  EXPECT_EQ(3, dates["WE"][0].day_.day());
  EXPECT_EQ(calendar_date::ADD, dates["WE"][0].type_);

  EXPECT_EQ(2006, dates["WD"][1].day_.year());
  EXPECT_EQ(7, dates["WD"][1].day_.month());
  EXPECT_EQ(4, dates["WD"][1].day_.day());
  EXPECT_EQ(calendar_date::REMOVE, dates["WD"][1].type_);

  EXPECT_EQ(2006, dates["WE"][1].day_.year());
  EXPECT_EQ(7, dates["WE"][1].day_.month());
  EXPECT_EQ(4, dates["WE"][1].day_.day());
  EXPECT_EQ(calendar_date::ADD, dates["WE"][1].type_);
}
