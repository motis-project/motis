#include "gtest/gtest.h"

#include "motis/loader/gtfs/calendar.h"
#include "motis/loader/gtfs/calendar_date.h"
#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/services.h"

#include "./resources.h"

using namespace utl;
using motis::bitfield;
using namespace motis::loader;
using namespace motis::loader::gtfs;

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
  auto const dates = read_calendar_date(
      loaded_file{SCHEDULES / "example" / CALENDAR_DATES_FILE});
  auto const calendar =
      read_calendar(loaded_file{SCHEDULES / "example" / CALENDAR_FILE});
  auto const traffic_days = merge_traffic_days(calendar, dates);

  auto we_bit_str = std::string{"1111000110000011000001100000110"};
  auto wd_bit_str = std::string{"0000111001111100111110011111001"};
  std::reverse(begin(we_bit_str), end(we_bit_str));
  std::reverse(begin(wd_bit_str), end(wd_bit_str));

  auto const we_traffic_days = bitfield{we_bit_str};
  auto const wd_traffic_days = bitfield{wd_bit_str};

  EXPECT_EQ(boost::gregorian::date(2006, 7, 1), traffic_days.first_day_);
  EXPECT_EQ(boost::gregorian::date(2006, 7, 31), traffic_days.last_day_);

  EXPECT_EQ(we_traffic_days, *traffic_days.traffic_days_.at("WE"));
  EXPECT_EQ(wd_traffic_days, *traffic_days.traffic_days_.at("WD"));
}
