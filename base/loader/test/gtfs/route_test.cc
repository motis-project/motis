#include "gtest/gtest.h"

#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/route.h"

#include "./resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;

TEST(loader_gtfs_route, read_routes_example_data) {
  auto agencies =
      read_agencies(loaded_file{SCHEDULES / "example" / AGENCY_FILE});
  auto routes =
      read_routes(loaded_file{SCHEDULES / "example" / ROUTES_FILE}, agencies);

  EXPECT_EQ(1, routes.size());
  EXPECT_NE(end(routes), routes.find("A"));
  EXPECT_EQ(nullptr, routes["A"]->agency_);
  EXPECT_EQ("17", routes["A"]->short_name_);
  EXPECT_EQ("Mission", routes["A"]->long_name_);
  EXPECT_EQ(3, routes["A"]->type_);
}

TEST(loader_gtfs_route, read_routes_berlin_data) {
  auto agencies =
      read_agencies(loaded_file{SCHEDULES / "berlin" / AGENCY_FILE});
  auto routes =
      read_routes(loaded_file{SCHEDULES / "berlin" / ROUTES_FILE}, agencies);

  EXPECT_EQ(8, routes.size());

  ASSERT_NE(end(routes), routes.find("1"));
  EXPECT_EQ("ANG---", routes["1"]->agency_->id_);
  EXPECT_EQ("SXF2", routes["1"]->short_name_);
  EXPECT_EQ("", routes["1"]->long_name_);
  EXPECT_EQ(700, routes["1"]->type_);

  ASSERT_NE(end(routes), routes.find("809"));
  EXPECT_EQ("N04---", routes["809"]->agency_->id_);
  EXPECT_EQ("", routes["809"]->short_name_);
  EXPECT_EQ("Leisnig -- Leipzig, Hauptbahnhof", routes["809"]->long_name_);
  EXPECT_EQ(100, routes["809"]->type_);

  ASSERT_NE(end(routes), routes.find("812"));
  EXPECT_EQ("N04---", routes["812"]->agency_->id_);
  EXPECT_EQ("RB14", routes["812"]->short_name_);
  EXPECT_EQ("", routes["812"]->long_name_);
  EXPECT_EQ(100, routes["812"]->type_);
}
