#include "gtest/gtest.h"

#include "motis/config.h"

using namespace motis;
using namespace std::string_literals;

TEST(motis, config) {
  auto const c = config{
      .osm_ = {"europe-latest.osm.pbf"},
      .tiles_ = {{.profile_ = "deps/tiles/profile/profile.lua"}},
      .timetable_ = {config::timetable{
          .first_day_ = "2024-10-02",
          .num_days_ = 2U,
          .datasets_ =
              {{"de",
                {.path_ = "delfi.gtfs.zip",
                 .clasz_bikes_allowed_ = {{{"LONG_DISTANCE", false},
                                           {"REGIONAL_FAST", true}}},
                 .rt_ =
                     {{{.url_ =
                            R"(https://stc.traines.eu/mirror/german-delfi-gtfs-rt/latest.gtfs-rt.pbf)",
                        .headers_ = {{{"Authorization", "test"}}}}}}}},
               {"nl",
                {.path_ = "nl.gtfs.zip",
                 .rt_ =
                     {{{.url_ = R"(https://gtfs.ovapi.nl/nl/trainUpdates.pb)"},
                       {.url_ =
                            R"(https://gtfs.ovapi.nl/nl/tripUpdates.pb)"}}}}}},
          .assistance_times_ = {"assistance.csv"}}},
      .street_routing_ = true,
      .limits_ = config::limits{},
      .osr_footpath_ = true,
      .geocoding_ = true};

  EXPECT_EQ(fmt::format(R"(
osm: europe-latest.osm.pbf
tiles:
  profile: deps/tiles/profile/profile.lua
  db_size: 274877906944
  flush_threshold: 100000
timetable:
  first_day: 2024-10-02
  num_days: 2
  tb: false
  railviz: true
  with_shapes: true
  adjust_footpaths: true
  merge_dupes_intra_src: false
  merge_dupes_inter_src: false
  link_stop_distance: 100
  update_interval: 60
  http_timeout: 30
  canned_rt: false
  incremental_rt_update: false
  use_osm_stop_coordinates: false
  extend_missing_footpaths: false
  max_footpath_length: 15
  max_matching_distance: 25.000000
  preprocess_max_matching_distance: 250.000000
  datasets:
    de:
      path: delfi.gtfs.zip
      default_bikes_allowed: false
      default_cars_allowed: false
      extend_calendar: false
      clasz_bikes_allowed:
        LONG_DISTANCE: false
        REGIONAL_FAST: true
      rt:
        - url: https://stc.traines.eu/mirror/german-delfi-gtfs-rt/latest.gtfs-rt.pbf
          headers:
            Authorization: test
          protocol: gtfsrt
    nl:
      path: nl.gtfs.zip
      default_bikes_allowed: false
      default_cars_allowed: false
      extend_calendar: false
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
          protocol: gtfsrt
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
          protocol: gtfsrt
  assistance_times: assistance.csv
elevators: false
street_routing: true
limits:
  stoptimes_max_results: 256
  plan_max_results: 256
  plan_max_search_window_minutes: 5760
  stops_max_results: 2048
  onetoall_max_results: 65535
  onetoall_max_travel_minutes: 90
  routing_max_timeout_seconds: 90
  gtfsrt_expose_max_trip_updates: 100
  street_routing_max_prepost_transit_seconds: 3600
  street_routing_max_direct_seconds: 21600
osr_footpath: true
geocoding: true
reverse_geocoding: false
)",
                        std::thread::hardware_concurrency()),
            (std::stringstream{} << "\n"
                                 << c << "\n")
                .str());

  EXPECT_EQ(c, config::read(R"(
osm: europe-latest.osm.pbf
tiles:
  profile: deps/tiles/profile/profile.lua
timetable:
  first_day: 2024-10-02
  num_days: 2
  tb: false
  datasets:
    de:
      path: delfi.gtfs.zip
      clasz_bikes_allowed:
        LONG_DISTANCE: false
        REGIONAL_FAST: true
      rt:
        - url: https://stc.traines.eu/mirror/german-delfi-gtfs-rt/latest.gtfs-rt.pbf
          headers:
            Authorization: test
    nl:
      path: nl.gtfs.zip
      default_bikes_allowed: false
      default_cars_allowed: false
      extend_calendar: false
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
  assistance_times: assistance.csv
elevators: false
street_routing: true
osr_footpath: true
geocoding: true
)"s));

  EXPECT_TRUE(c.use_street_routing());

  // Using street_routing struct
  {
    // Setting height_data_dir
    {
      auto const street_routing_config =
          config{.osm_ = {"europe-latest.osm.pbf"},
                 .street_routing_ =
                     config::street_routing{.elevation_data_dir_ = "srtm/"},
                 .limits_ = config::limits{}};
      EXPECT_EQ(street_routing_config, config::read(R"(
street_routing:
  elevation_data_dir: srtm/
osm: europe-latest.osm.pbf
)"s));
      EXPECT_TRUE(street_routing_config.use_street_routing());
    }

    // Using empty street_routing map
    {
      auto const street_routing_config =
          config{.osm_ = {"europe-latest.osm.pbf"},
                 .street_routing_ = config::street_routing{},
                 .limits_ = config::limits{}};
      EXPECT_EQ(street_routing_config, config::read(R"(
street_routing: {}
osm: europe-latest.osm.pbf
)"s));
      EXPECT_TRUE(street_routing_config.use_street_routing());
    }

    // No street_routing defined
    EXPECT_FALSE(config::read(R"(
osm: europe-latest.osm.pbf
)"s)
                     .use_street_routing());

    // street_routing disabled
    EXPECT_FALSE(config::read(R"(
osm: europe-latest.osm.pbf
street_routing: false
)"s)
                     .use_street_routing());

    // Will throw if street_routing is set but osm is not
    EXPECT_ANY_THROW(config::read(R"(
street_routing: {}
)"s));
  }
}
