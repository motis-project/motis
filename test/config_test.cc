#include <array>
#include <string_view>
#include <utility>

#include "fmt/format.h"

#include "gtest/gtest.h"

#include "motis/config.h"

using namespace motis;
using namespace std::string_literals;
using namespace std::string_view_literals;

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
            c.vehicle_eta_mode("any-feed", nigiri::clasz::kBus));
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
            c.vehicle_eta_mode("A", nigiri::clasz::kBus));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::shadow,
            c.vehicle_eta_mode("A", nigiri::clasz::kTram));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::effective,
            c.vehicle_eta_mode("B", nigiri::clasz::kBus));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::shadow,
            c.vehicle_eta_mode("B", nigiri::clasz::kTram));
}

TEST(motis, vehicle_eta_accepts_canonical_transit_modes_and_aliases) {
  using nigiri::clasz;

  constexpr auto modes = std::array{
      std::pair{"AIRPLANE"sv, clasz::kAir},
      std::pair{"AIR"sv, clasz::kAir},
      std::pair{"HIGHSPEED_RAIL"sv, clasz::kHighSpeed},
      std::pair{"HIGHSPEED"sv, clasz::kHighSpeed},
      std::pair{"LONG_DISTANCE"sv, clasz::kLongDistance},
      std::pair{"LONGDISTANCE"sv, clasz::kLongDistance},
      std::pair{"COACH"sv, clasz::kCoach},
      std::pair{"NIGHT_RAIL"sv, clasz::kNight},
      std::pair{"NIGHT"sv, clasz::kNight},
      std::pair{"RIDE_SHARING"sv, clasz::kRideSharing},
      std::pair{"REGIONAL_FAST_RAIL"sv, clasz::kRegional},
      std::pair{"REGIONAL_RAIL"sv, clasz::kRegional},
      std::pair{"REGIONALFAST"sv, clasz::kRegional},
      std::pair{"REGIONAL"sv, clasz::kRegional},
      std::pair{"SUBURBAN"sv, clasz::kSuburban},
      std::pair{"METRO"sv, clasz::kSuburban},
      std::pair{"SUBWAY"sv, clasz::kSubway},
      std::pair{"TRAM"sv, clasz::kTram},
      std::pair{"BUS"sv, clasz::kBus},
      std::pair{"FERRY"sv, clasz::kShip},
      std::pair{"SHIP"sv, clasz::kShip},
      std::pair{"ODM"sv, clasz::kODM},
      std::pair{"FUNICULAR"sv, clasz::kFunicular},
      std::pair{"CABLE_CAR"sv, clasz::kFunicular},
      std::pair{"AERIAL_LIFT"sv, clasz::kAerialLift},
      std::pair{"AREAL_LIFT"sv, clasz::kAerialLift},
      std::pair{"OTHER"sv, clasz::kOther},
  };

  for (auto const& [configured_mode, query_clasz] : modes) {
    auto const c = config::read(fmt::format(R"(
timetable:
  datasets:
    A:
      path: a.gtfs.zip
  vehicle_eta:
    modes:
      {}: effective
)",
                                            configured_mode));
    EXPECT_EQ(config::timetable::vehicle_eta::mode::effective,
              c.vehicle_eta_mode("A", query_clasz))
        << configured_mode;
  }
}

TEST(motis, vehicle_eta_feed_selectors_normalize_aliases) {
  auto const c = config::read(R"(
timetable:
  datasets:
    A:
      path: a.gtfs.zip
  vehicle_eta:
    feeds:
      A:
        modes: [SHIP]
        mode: effective
)"s);

  EXPECT_TRUE(c.vehicle_eta_enabled());
  EXPECT_EQ(config::timetable::vehicle_eta::mode::effective,
            c.vehicle_eta_mode("A", nigiri::clasz::kShip));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::off,
            c.vehicle_eta_mode("A", nigiri::clasz::kBus));
}

TEST(motis, vehicle_eta_off_policies_do_not_enable_work) {
  auto const c = config::read(R"(
timetable:
  datasets:
    A:
      path: a.gtfs.zip
    B:
      path: b.gtfs.zip
  vehicle_eta:
    mode: effective
    modes:
      BUS: shadow
    feeds:
      A:
        mode: off
      B:
        mode: off
)"s);

  EXPECT_FALSE(c.vehicle_eta_enabled());
  EXPECT_EQ(config::timetable::vehicle_eta::mode::off,
            c.vehicle_eta_mode("A", nigiri::clasz::kBus));
  EXPECT_EQ(config::timetable::vehicle_eta::mode::off,
            c.vehicle_eta_mode("B", nigiri::clasz::kTram));
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
  EXPECT_ANY_THROW(config::read(config_with(R"(    feeds:
      A:
        modes: []
        mode: shadow)")));
  EXPECT_ANY_THROW(config::read(config_with(R"(    modes:
      AIR: shadow
      AIRPLANE: effective)")));
  EXPECT_ANY_THROW(config::read(config_with(R"(    feeds:
      A:
        modes: [SHIP, FERRY]
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
                 .clasz_bikes_allowed_ = {{{"LONGDISTANCE", false},
                                           {"REGIONAL", true}}},
                 .clasz_reservation_not_required_ = {{{"LONGDISTANCE", true},
                                                      {"COACH", false}}},
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
  extend_missing_footpaths: false
  max_footpath_length: 15
  default_transfer_time: 2
  max_matching_distance: 25.000000
  preprocess_max_matching_distance: 250.000000
  datasets:
    de:
      path: delfi.gtfs.zip
      extend_calendar: false
      default_bikes_allowed: false
      default_cars_allowed: false
      default_reservation_not_required: true
      clasz_bikes_allowed:
        LONGDISTANCE: false
        REGIONAL: true
      clasz_reservation_not_required:
        COACH: false
        LONGDISTANCE: true
      rt:
        - url: https://stc.traines.eu/mirror/german-delfi-gtfs-rt/latest.gtfs-rt.pbf
          headers:
            Authorization: test
          last_good_ttl: 180
          protocol: gtfsrt
    nl:
      path: nl.gtfs.zip
      extend_calendar: false
      default_bikes_allowed: false
      default_cars_allowed: false
      default_reservation_not_required: true
      clasz_reservation_not_required:
        AIR: false
        COACH: false
        NIGHT: false
        ODM: false
        RIDESHARING: false
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
  stoptimes_max_results: 1024
  plan_max_results: 256
  plan_max_search_window_minutes: 5760
  stops_max_results: 8192
  onetomany_max_many: 128
  onetoall_max_results: 65535
  onetoall_max_travel_minutes: 90
  routing_max_timeout_seconds: 90
  gtfsrt_expose_max_trip_updates: 100
  street_routing_max_prepost_transit_seconds: 3600
  street_routing_max_direct_seconds: 21600
  geocode_max_suggestions: 512
  reverse_geocode_max_results: 512
  max_max_matching_distance: 250.000000
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
        LONGDISTANCE: false
        REGIONAL: true
      clasz_reservation_not_required:
        COACH: false
        LONGDISTANCE: true
      rt:
        - url: https://stc.traines.eu/mirror/german-delfi-gtfs-rt/latest.gtfs-rt.pbf
          headers:
            Authorization: test
    nl:
      path: nl.gtfs.zip
      default_bikes_allowed: false
      default_cars_allowed: false
      default_reservation_not_required: true
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
