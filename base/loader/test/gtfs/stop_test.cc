#include "gtest/gtest.h"

#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/stop.h"
#include "motis/loader/util.h"

#include "./resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;

TEST(loader_gtfs_stop, read_stations_example_data) {
  auto stops = read_stops(loaded_file{SCHEDULES / "example" / STOPS_FILE});

  EXPECT_EQ(8, stops.size());

  auto s1_it = stops.find("S1");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("Mission St. & Silver Ave.", s1_it->second->name_);
  EXPECT_FLOAT_EQ(37.728631, s1_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(-122.431282, s1_it->second->coord_.lng_);

  auto s6_it = stops.find("S6");
  ASSERT_NE(s6_it, end(stops));
  EXPECT_EQ("Mission St. & 15th St.", s6_it->second->name_);
  EXPECT_FLOAT_EQ(37.766629, s6_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(-122.419782, s6_it->second->coord_.lng_);

  auto s8_it = stops.find("S8");
  ASSERT_NE(s8_it, end(stops));
  EXPECT_EQ("24th St. Mission Station", s8_it->second->name_);
  EXPECT_FLOAT_EQ(37.752240, s8_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(-122.418450, s8_it->second->coord_.lng_);
}

TEST(loader_gtfs_stop, read_stations_berlin_data) {
  auto stops = read_stops(loaded_file{SCHEDULES / "berlin" / STOPS_FILE});

  EXPECT_EQ(3, stops.size());

  auto s0_it = stops.find("5100071");
  ASSERT_NE(s0_it, end(stops));
  EXPECT_EQ("Zbaszynek", s0_it->second->name_);
  EXPECT_FLOAT_EQ(52.2425040, s0_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(15.8180870, s0_it->second->coord_.lng_);

  auto s1_it = stops.find("9230005");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("S Potsdam Hauptbahnhof Nord", s1_it->second->name_);
  EXPECT_FLOAT_EQ(52.3927320, s1_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(13.0668480, s1_it->second->coord_.lng_);

  auto s2_it = stops.find("9230006");
  ASSERT_NE(s2_it, end(stops));
  EXPECT_EQ("Potsdam, Charlottenhof Bhf", s2_it->second->name_);
  EXPECT_FLOAT_EQ(52.3930040, s2_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(13.0362980, s2_it->second->coord_.lng_);
}
