#include "gtest/gtest.h"

#include "motis/loader/gtfs/calendar.h"
#include "motis/loader/gtfs/calendar_date.h"
#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/services.h"

#include "./resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;
using namespace date;

/*
 * -- BASE --
 * WE:       11         WD:       00
 *      0000011              1111100
 *      0000011              1111100
 *      0000011              1111100
 *      0000011              1111100
 *      0                    1
 */

/*
 * -- DATES --
 * WE:       11         WD:       00
 *      1100011              0011100
 *      0000011              1111100
 *      0000011              1111100
 *      0000011              1111100
 *      0                    1
 */

TEST(loader_gtfs_traffic_days, read_traffic_days_example_data) {
  auto dates = read_calendar_date(
      loaded_file{SCHEDULES / "example" / CALENDAR_DATES_FILE});
  auto calendar =
      read_calendar(loaded_file{SCHEDULES / "example" / CALENDAR_FILE});
  auto traffic_days = merge_traffic_days(calendar, dates);

  std::string we_bit_str = "1111000110000011000001100000110";
  std::string wd_bit_str = "0000111001111100111110011111001";
  std::reverse(begin(we_bit_str), end(we_bit_str));
  std::reverse(begin(wd_bit_str), end(wd_bit_str));
  bitfield const we_traffic_days(we_bit_str);
  bitfield const wd_traffic_days(wd_bit_str);

  EXPECT_TRUE(2006_y / 7 / 1 == traffic_days.first_day_);
  EXPECT_TRUE(2006_y / 7 / 31 == traffic_days.last_day_);

  EXPECT_TRUE(we_traffic_days == *traffic_days.traffic_days_["WE"]);
  EXPECT_TRUE(wd_traffic_days == *traffic_days.traffic_days_["WD"]);
}
