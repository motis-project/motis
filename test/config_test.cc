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
      .osr_footpath_ = true,
      .geocoding_ = true};

  EXPECT_EQ(fmt::format(R"(
osm: europe-latest.osm.pbf
tiles:
  profile: deps/tiles/profile/profile.lua
  db_size: 1099511627776
  flush_threshold: 10000000
timetable:
  first_day: 2024-10-02
  num_days: 2
  railviz: true
  with_shapes: true
  ignore_errors: false
  adjust_footpaths: true
  merge_dupes_intra_src: false
  merge_dupes_inter_src: false
  link_stop_distance: 100
  update_interval: 60
  http_timeout: 10
  incremental_rt_update: false
  max_footpath_length: 15
  datasets:
    de:
      path: delfi.gtfs.zip
      default_bikes_allowed: false
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
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
  assistance_times: assistance.csv
street_routing: true
osr_footpath: true
elevators: false
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
      rt:
        - url: https://gtfs.ovapi.nl/nl/trainUpdates.pb
        - url: https://gtfs.ovapi.nl/nl/tripUpdates.pb
  assistance_times: assistance.csv
street_routing: true
osr_footpath: true
geocoding: true
)"s));
}