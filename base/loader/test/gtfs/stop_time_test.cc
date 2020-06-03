#include "gtest/gtest.h"

#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/stop_time.h"

#include "./resources.h"

using namespace utl;

namespace motis::loader::gtfs {

TEST(loader_gtfs_route, read_stop_times_example_data) {
  auto agencies =
      read_agencies(loaded_file{SCHEDULES / "example" / AGENCY_FILE});
  auto routes =
      read_routes(loaded_file{SCHEDULES / "example" / ROUTES_FILE}, agencies);
  auto dates = read_calendar_date(
      loaded_file{SCHEDULES / "example" / CALENDAR_DATES_FILE});
  auto calendar =
      read_calendar(loaded_file{SCHEDULES / "example" / CALENDAR_FILE});
  auto traffic_days = merge_traffic_days(calendar, dates);
  auto [trips, blocks] = read_trips(
      loaded_file{SCHEDULES / "example" / TRIPS_FILE}, routes, traffic_days);
  auto stops = read_stops(loaded_file{SCHEDULES / "example" / STOPS_FILE});
  read_stop_times(loaded_file{SCHEDULES / "example" / STOP_TIMES_FILE}, trips,
                  stops);

  auto awe1_it = trips.find("AWE1");
  ASSERT_NE(end(trips), awe1_it);

  auto& awe1_stops = awe1_it->second->stop_times_;
  auto& stop = awe1_stops[1];
  EXPECT_EQ("S1", stop.stop_->id_);
  EXPECT_EQ(6, stop.arr_.time_);
  EXPECT_EQ(6, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[2];
  EXPECT_EQ("S2", stop.stop_->id_);
  EXPECT_EQ(-1, stop.arr_.time_);
  EXPECT_EQ(-1, stop.dep_.time_);
  EXPECT_FALSE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[3];
  EXPECT_EQ("S3", stop.stop_->id_);
  EXPECT_EQ(6, stop.arr_.time_);
  EXPECT_EQ(6, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[4];
  EXPECT_EQ("S5", stop.stop_->id_);
  EXPECT_EQ(-1, stop.arr_.time_);
  EXPECT_EQ(-1, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[5];
  EXPECT_EQ("S6", stop.stop_->id_);
  EXPECT_EQ(6, stop.arr_.time_);
  EXPECT_EQ(6, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);
}

}  // namespace motis::loader::gtfs
