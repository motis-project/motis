#include "gtest/gtest.h"

#include "motis/config.h"

using namespace motis;
using namespace std::string_literals;

TEST(motis, vehicle_eta_defaults_to_off_without_history_work) {
  auto const c = config::read(R"(
timetable:
  datasets:
    A:
      path: a.gtfs.zip
)"s);

  ASSERT_TRUE(c.timetable_.has_value());
  EXPECT_FALSE(c.timetable_->vehicle_eta_.has_value());
  EXPECT_EQ(config::timetable::vehicle_eta::mode::off,
            c.vehicle_eta_mode("any-feed", "BUS"));
  EXPECT_FALSE(c.vehicle_eta_enabled());
}

TEST(motis, vehicle_eta_parses_modes_history_and_overrides) {
  auto const c = config::read(R"(
timetable:
  datasets:
    A:
      path: a.gtfs.zip
    B:
      path: b.gtfs.zip
  vehicle_eta:
    mode: shadow
    history:
      max_age_seconds: 300
      max_observations_per_vehicle: 20
    modes:
      BUS: effective
    feeds:
      A:
        modes: [BUS]
        mode: off
)"s);

  ASSERT_TRUE(c.timetable_->vehicle_eta_.has_value());
  EXPECT_EQ(300, c.timetable_->vehicle_eta_->history_.max_age_seconds_);
  EXPECT_EQ(20U,
            c.timetable_->vehicle_eta_->history_.max_observations_per_vehicle_);
  EXPECT_TRUE(c.vehicle_eta_enabled());
  EXPECT_EQ(config::timetable::vehicle_eta::mode::off,
            c.vehicle_eta_mode("A", "BUS"));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::shadow,
            c.vehicle_eta_mode("A", "TRAM"));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::effective,
            c.vehicle_eta_mode("B", "BUS"));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::shadow,
            c.vehicle_eta_mode("B", "TRAM"));
}

TEST(motis, vehicle_eta_rejects_invalid_configuration) {
  auto const config_with = [](std::string_view const vehicle_eta) {
    return fmt::format(R"(
timetable:
  datasets:
    A:
      path: a.gtfs.zip
  vehicle_eta:
{}
)",
                       vehicle_eta);
  };

  EXPECT_ANY_THROW(config::read(config_with("    mode: invalid")));
  EXPECT_ANY_THROW(config::read(config_with(R"(    feeds:
      missing:
        mode: shadow)")));
  EXPECT_ANY_THROW(config::read(config_with(R"(    modes:
      INVALID: shadow)")));
  EXPECT_ANY_THROW(config::read(config_with(R"(    feeds:
      A:
        modes: [INVALID]
        mode: shadow)")));
  EXPECT_ANY_THROW(
      config::read(config_with("    history:\n      max_age_seconds: -1")));
  EXPECT_ANY_THROW(
      config::read(config_with("    history:\n      max_age_seconds: 0")));
  EXPECT_ANY_THROW(config::read(
      config_with("    history:\n      max_observations_per_vehicle: 0")));
}

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
          last_good_ttl: 180
          protocol: gtfsrt
    nl:
      path: nl.gtfs.zip
      default_bikes_allowed: false
      default_cars_allowed: false
      extend_calendar: false
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
          last_good_ttl: 180
          protocol: gtfsrt
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
          last_good_ttl: 180
          protocol: gtfsrt
  assistance_times: assistance.csv
elevators: false
street_routing: true
limits:
  stoptimes_max_results: 256
  plan_max_results: 256
  plan_max_search_window_minutes: 5760
  stops_max_results: 2048
  onetomany_max_many: 128
  onetoall_max_results: 65535
  onetoall_max_travel_minutes: 90
  routing_max_timeout_seconds: 90
  gtfsrt_expose_max_trip_updates: 100
  street_routing_max_prepost_transit_seconds: 3600
  street_routing_max_direct_seconds: 21600
  geocode_max_suggestions: 10
  reverse_geocode_max_results: 5
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

  EXPECT_ANY_THROW(config::read(R"(
timetable:
  datasets:
    test:
      path: test.gtfs.zip
      rt:
        - url: https://example.test/trip_updates
          last_good_ttl: 0
          protocol: gtfsrt
)"s));

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
