#include "gtest/gtest.h"

#include <algorithm>

#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/trip.h"

#include "./resources.h"

using namespace utl;
using namespace motis::loader;

namespace motis::loader::gtfs {

TEST(loader_gtfs_trip, read_trips_example_data) {
  auto agencies =
      read_agencies(loaded_file{SCHEDULES / "example" / AGENCY_FILE});
  auto routes =
      read_routes(loaded_file{SCHEDULES / "example" / ROUTES_FILE}, agencies);
  auto dates = read_calendar_date(
      loaded_file{SCHEDULES / "example" / CALENDAR_DATES_FILE});
  auto calendar =
      read_calendar(loaded_file{SCHEDULES / "example" / CALENDAR_FILE});
  auto services = merge_traffic_days(calendar, dates);
  auto [trips, blocks] = read_trips(
      loaded_file{SCHEDULES / "example" / TRIPS_FILE}, routes, services);

  EXPECT_EQ(2, trips.size());
  EXPECT_NE(end(trips), trips.find("AWE1"));
  EXPECT_EQ("A", trips["AWE1"]->route_->id_);
  EXPECT_EQ("Downtown", trips["AWE1"]->headsign_);
}

TEST(loader_gtfs_trip, read_trips_berlin_data) {
  auto agencies =
      read_agencies(loaded_file{SCHEDULES / "berlin" / AGENCY_FILE});
  auto routes =
      read_routes(loaded_file{SCHEDULES / "berlin" / ROUTES_FILE}, agencies);
  auto dates = read_calendar_date(
      loaded_file{SCHEDULES / "berlin" / CALENDAR_DATES_FILE});
  auto calendar =
      read_calendar(loaded_file{SCHEDULES / "berlin" / CALENDAR_FILE});
  auto services = merge_traffic_days(calendar, dates);
  auto [trips, blocks] = read_trips(
      loaded_file{SCHEDULES / "berlin" / TRIPS_FILE}, routes, services);

  EXPECT_EQ(3, trips.size());

  EXPECT_NE(end(trips), trips.find("1"));
  EXPECT_EQ("1", trips["1"]->route_->id_);
  // EXPECT_EQ("000856", trips["1"].service_id);
  EXPECT_EQ("Flughafen Schönefeld Terminal (Airport)", trips["1"]->headsign_);

  EXPECT_NE(end(trips), trips.find("2"));
  EXPECT_EQ("1", trips["2"]->route_->id_);
  // EXPECT_EQ("000856", trips["2"].service_id);
  EXPECT_EQ("S Potsdam Hauptbahnhof", trips["2"]->headsign_);

  EXPECT_NE(end(trips), trips.find("3"));
  EXPECT_EQ("2", trips["3"]->route_->id_);
  // EXPECT_EQ("000861", trips["3"].service_id);
  EXPECT_EQ("Golzow (PM), Schule", trips["3"]->headsign_);
}

}  // namespace motis::loader::gtfs
