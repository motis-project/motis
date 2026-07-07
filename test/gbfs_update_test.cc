#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "boost/json.hpp"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"

#include "fmt/format.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"
#include "motis/metrics_registry.h"

namespace fs = std::filesystem;
namespace json = boost::json;
using namespace motis;
using namespace motis::gbfs;
using namespace std::string_view_literals;

namespace {

constexpr auto const kBikeType = "bike"sv;
constexpr auto const kScooterType = "scooter"sv;
constexpr auto const kStation1 = "station-1"sv;

fs::path make_temp_dir(std::string_view const name) {
  auto dir = fs::temp_directory_path() / "motis-gbfs-tests" / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void write_file(fs::path const& path, std::string_view const content) {
  fs::create_directories(path.parent_path());
  auto out = std::ofstream{path};
  out << content;
}

void write_discovery(fs::path const& dir,
                     std::string_view vehicle_feed = "free_bike_status",
                     bool const include_geofencing = true,
                     bool const include_vehicle_types = true) {
  auto feeds = json::array{
      {{"name", "system_information"}, {"url", "system_information"}},
      {{"name", "station_information"}, {"url", "station_information"}},
      {{"name", "station_status"}, {"url", "station_status"}},
  };
  if (include_vehicle_types) {
    feeds.push_back({{"name", "vehicle_types"}, {"url", "vehicle_types"}});
  }
  if (!vehicle_feed.empty()) {
    feeds.push_back({{"name", vehicle_feed}, {"url", vehicle_feed}});
  }
  if (include_geofencing) {
    feeds.push_back(
        {{"name", "geofencing_zones"}, {"url", "geofencing_zones"}});
  }

  write_file(dir / "gbfs.json", json::serialize(json::value{
                                    {"last_updated", 1},
                                    {"ttl", 0},
                                    {"version", "2.3"},
                                    {"data", {{"feeds", std::move(feeds)}}},
                                }));
}

void write_system_information(fs::path const& dir,
                              std::string_view const name = "Test Bikes",
                              std::string_view const color = "#112233") {
  write_file(dir / "system_information.json",
             json::serialize(json::value{
                 {"last_updated", 10},
                 {"ttl", 0},
                 {"version", "2.3"},
                 {"data",
                  {{"system_id", "test-system"},
                   {"name", name},
                   {"brand_assets", {{"color", color}}}}},
             }));
}

void write_vehicle_types(
    fs::path const& dir,
    bool const include_scooter = false,
    std::string_view const bike_return_constraint = "hybrid") {
  auto types = json::array{};
  types.push_back({{"vehicle_type_id", "bike"},
                   {"form_factor", "bicycle"},
                   {"propulsion_type", "human"},
                   {"return_constraint", bike_return_constraint}});
  if (include_scooter) {
    types.push_back({{"vehicle_type_id", "scooter"},
                     {"form_factor", "scooter_standing"},
                     {"propulsion_type", "electric"},
                     {"return_constraint", "free_floating"}});
  }

  write_file(dir / "vehicle_types.json",
             json::serialize(json::value{
                 {"last_updated", 20},
                 {"ttl", 0},
                 {"version", "3.0"},
                 {"data", {{"vehicle_types", std::move(types)}}},
             }));
}

void write_station_information(fs::path const& dir,
                               double const station_1_lat = 49.871651,
                               double const station_1_lon = 8.631084) {
  write_file(dir / "station_information.json",
             json::serialize(json::value{
                 {"last_updated", 30},
                 {"ttl", 0},
                 {"version", "2.3"},
                 {"data",
                  {{"stations", json::array{{{"station_id", "station-1"},
                                             {"name", "Main Station"},
                                             {"lat", station_1_lat},
                                             {"lon", station_1_lon}}}}}},
             }));
}

void write_station_information_with_area(fs::path const& dir) {
  write_file(dir / "station_information.json", R"({
    "last_updated": 30,
    "ttl": 0,
    "version": "2.3",
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "name": "Main Station",
          "lat": 49.871651,
          "lon": 8.631084,
          "station_area": {
            "type": "MultiPolygon",
            "coordinates": [[[
              [8.626, 49.870],
              [8.634, 49.870],
              [8.634, 49.877],
              [8.626, 49.877],
              [8.626, 49.870]
            ]]]
          }
        }
      ]
    }
  })");
}

void write_station_status(fs::path const& dir,
                          unsigned const bike_count = 3U,
                          bool const include_bad_station = false,
                          std::string_view const type_id = kBikeType,
                          unsigned const last_updated = 40U,
                          unsigned const ttl = 0U) {
  auto stations = json::array{};
  stations.push_back({{"station_id", "station-1"},
                      {"num_bikes_available", bike_count},
                      {"vehicle_types_available",
                       {{{"vehicle_type_id", type_id}, {"count", bike_count}}}},
                      {"vehicle_docks_available",
                       {{{"vehicle_type_ids", {type_id}}, {"count", 2U}}}},
                      {"is_renting", true},
                      {"is_returning", true}});
  if (include_bad_station) {
    stations.push_back(
        {{"station_id", "missing-station"}, {"num_bikes_available", 9U}});
  }

  write_file(dir / "station_status.json",
             json::serialize(json::value{
                 {"last_updated", last_updated},
                 {"ttl", ttl},
                 {"version", "2.3"},
                 {"data", {{"stations", std::move(stations)}}},
             }));
}

void write_free_bike_status(fs::path const& dir,
                            std::string_view const vehicle_id = "vehicle-1",
                            double const lat = 49.875309,
                            double const lon = 8.627667,
                            bool const reserved = false,
                            bool const disabled = false,
                            std::string_view const type_id = kBikeType) {
  write_file(dir / "free_bike_status.json",
             json::serialize(json::value{
                 {"last_updated", 50},
                 {"ttl", 0},
                 {"version", "2.3"},
                 {"data",
                  {{"bikes", json::array{{{"bike_id", vehicle_id},
                                          {"lat", lat},
                                          {"lon", lon},
                                          {"is_reserved", reserved},
                                          {"is_disabled", disabled},
                                          {"vehicle_type_id", type_id}}}}}},
             }));
}

void write_vehicle_status_at_station(fs::path const& dir,
                                     std::string_view const station_id,
                                     std::string_view const type_id) {
  write_file(dir / "vehicle_status.json",
             json::serialize(json::value{
                 {"last_updated", 50},
                 {"ttl", 0},
                 {"version", "3.0"},
                 {"data",
                  {{"vehicles", json::array{{{"vehicle_id", "docked-vehicle"},
                                             {"station_id", station_id},
                                             {"is_reserved", false},
                                             {"is_disabled", false},
                                             {"vehicle_type_id", type_id}}}}}},
             }));
}

void write_empty_geofencing(fs::path const& dir) {
  write_file(
      dir / "geofencing_zones.json",
      json::serialize(json::value{
          {"last_updated", 60},
          {"ttl", 0},
          {"version", "2.3"},
          {"data",
           {{"geofencing_zones",
             {{"type", "FeatureCollection"}, {"features", json::array{}}}}}},
      }));
}

void write_geofencing_no_start_zone(fs::path const& dir) {
  write_file(dir / "geofencing_zones.json", R"({
    "last_updated": 60,
    "ttl": 0,
    "version": "3.0",
    "data": {
      "geofencing_zones": {
        "type": "FeatureCollection",
        "features": [
          {
            "type": "Feature",
            "properties": {
              "name": "No Start",
              "rules": [
                {
                  "vehicle_type_ids": ["bike"],
                  "ride_allowed": true,
                  "ride_start_allowed": false
                }
              ]
            },
            "geometry": {
              "type": "MultiPolygon",
              "coordinates": [[[
                [8.62, 49.87],
                [8.64, 49.87],
                [8.64, 49.88],
                [8.62, 49.88],
                [8.62, 49.87]
              ]]]
            }
          }
        ]
      },
      "global_rules": [
        {"vehicle_type_ids": ["bike"], "ride_allowed": true}
      ]
    }
  })");
}

void write_geofencing_global_no_return(fs::path const& dir) {
  write_file(dir / "geofencing_zones.json", R"({
    "last_updated": 60,
    "ttl": 0,
    "version": "3.0",
    "data": {
      "geofencing_zones": {
        "type": "FeatureCollection",
        "features": []
      },
      "global_rules": [
        {
          "vehicle_type_ids": ["bike"],
          "ride_allowed": false,
          "ride_through_allowed": true
        }
      ]
    }
  })");
}

void write_geofencing_unknown_type_ref(fs::path const& dir) {
  write_file(dir / "geofencing_zones.json", R"({
    "last_updated": 60,
    "ttl": 0,
    "version": "3.0",
    "data": {
      "geofencing_zones": {
        "type": "FeatureCollection",
        "features": [
          {
            "type": "Feature",
            "properties": {
              "name": "Unknown Type Zone",
              "rules": [
                {
                  "vehicle_type_ids": ["unknown-type"],
                  "ride_allowed": false
                }
              ]
            },
            "geometry": {
              "type": "MultiPolygon",
              "coordinates": [[[
                [8.62, 49.87],
                [8.64, 49.87],
                [8.64, 49.88],
                [8.62, 49.88],
                [8.62, 49.87]
              ]]]
            }
          }
        ]
      }
    }
  })");
}

void write_default_feed(fs::path const& dir) {
  write_discovery(dir);
  write_system_information(dir);
  write_vehicle_types(dir);
  write_station_information(dir);
  write_station_status(dir);
  write_free_bike_status(dir);
  write_empty_geofencing(dir);
}

void write_manifest(fs::path const& dir,
                    std::vector<std::string_view> const& provider_ids) {
  auto datasets = json::array{};
  for (auto const provider_id : provider_ids) {
    datasets.push_back(
        {{"system_id", provider_id},
         {"versions", {{{"version", "2.3"}, {"url", provider_id}}}}});
  }

  auto const manifest = json::serialize(json::value{
      {"last_updated", 1},
      {"ttl", 0},
      {"version", "3.0"},
      {"data", {{"datasets", std::move(datasets)}}},
  });
  write_file(dir / "gbfs.json", manifest);
  write_file(dir / "manifest.json", manifest);
}

data& street_data() {
  static auto d = []() {
    auto const path = fs::path{"test/data_gbfs_update_osm"};
    auto ec = std::error_code{};
    fs::remove_all(path, ec);
    auto const c = config{.osm_ = {"test/resources/test_case.osm.pbf"},
                          .street_routing_ = true};
    import(c, path);
    return std::make_unique<data>(path, c);
  }();
  return *d;
}

config make_gbfs_config(fs::path const& dir, std::string const& id = "test") {
  auto c = config{};
  c.gbfs_ = config::gbfs{
      .feeds_ = {{id,
                  config::gbfs::feed{
                      .url_ = dir.string(),
                      .ttl_ =
                          config::gbfs::ttl{.overwrite_ =
                                                std::map<std::string, unsigned>{
                                                    {"gbfs", 0U},
                                                    {"manifest", 0U},
                                                    {"system_information", 0U},
                                                    {"vehicle_types", 0U},
                                                    {"station_information", 0U},
                                                    {"station_status", 0U},
                                                    {"vehicle_status", 0U},
                                                    {"free_bike_status", 0U},
                                                    {"geofencing_zones",
                                                     0U}}}}}},
      .cache_size_ = 20U};
  return c;
}

config make_gbfs_config_without_ttl_overwrite(fs::path const& dir,
                                              std::string const& id = "test") {
  auto c = config{};
  c.gbfs_ =
      config::gbfs{.feeds_ = {{id, config::gbfs::feed{.url_ = dir.string()}}},
                   .cache_size_ = 20U};
  return c;
}

void run_update(config const& c, std::shared_ptr<gbfs_data>& gbfs) {
  auto& d = street_data();
  auto ioc = boost::asio::io_context{};
  auto metrics = metrics_registry{};
  boost::asio::co_spawn(
      ioc,
      [&]() -> boost::asio::awaitable<void> {
        co_await update(c, *d.w_, *d.l_, gbfs, &metrics);
      },
      boost::asio::detached);
  ioc.run();
}

provider_routing_data const& routing_data_for(gbfs_data& gbfs,
                                              gbfs_provider const& p) {
  auto& d = street_data();
  return *gbfs.get_products_routing_data(
                  *d.w_, *d.l_,
                  gbfs_products_ref{p.idx_, gbfs_products_idx_t{0}})
              ->provider_routing_data_;
}

std::size_t additional_node_count(gbfs_data& gbfs, gbfs_provider const& p) {
  auto const& prd = routing_data_for(gbfs, p);
  if (prd.products_.empty()) {
    return 0U;
  }
  return prd.products_.front().additional_nodes_.size();
}

bool rtree_contains(gbfs_data const& gbfs,
                    geo::latlng const& pos,
                    gbfs_provider_idx_t const idx,
                    double const radius = 5.0) {
  auto found = false;
  gbfs.provider_rtree_.in_radius(pos, radius, [&](gbfs_provider_idx_t const x) {
    found = found || x == idx;
  });
  return found;
}

std::size_t count_additional_vehicle_nodes(provider_routing_data const& prd) {
  if (prd.products_.empty()) {
    return 0U;
  }
  auto const& nodes = prd.products_.front().additional_nodes_;
  return static_cast<std::size_t>(
      std::count_if(begin(nodes), end(nodes), [](additional_node const& n) {
        return std::holds_alternative<additional_node::vehicle>(n.data_);
      }));
}

}  // namespace

TEST(motis, gbfs_update_initial_file_feed_creates_provider_state) {
  auto const dir = make_temp_dir("initial");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  ASSERT_NE(nullptr, gbfs);
  ASSERT_EQ(1U, gbfs->providers_.size());
  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));

  EXPECT_EQ("test", p.id_);
  EXPECT_EQ("test-system", p.sys_info_.id_);
  EXPECT_EQ("Test Bikes", p.sys_info_.name_);
  ASSERT_TRUE(p.color_.has_value());
  EXPECT_EQ("#112233", *p.color_);
  ASSERT_EQ(1U, p.stations_.size());
  EXPECT_EQ(
      3U,
      p.stations_.at(std::string{kStation1}).status_.num_vehicles_available_);
  ASSERT_EQ(1U, p.vehicle_status_.size());
  EXPECT_EQ("vehicle-1", p.vehicle_status_.front().id_);
  ASSERT_EQ(1U, p.products_.size());
  EXPECT_TRUE(p.has_vehicles_to_rent_);
  EXPECT_TRUE(gbfs->cache_.contains(p.idx_));
  EXPECT_GT(additional_node_count(*gbfs, p), 0U);
}

TEST(motis, gbfs_update_reuses_provider_and_applies_status_update) {
  auto const dir = make_temp_dir("status-update");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);
  auto const first_provider_idx =
      gbfs->providers_.at(gbfs->provider_by_id_.at("test"))->idx_;
  auto const first_vehicle_type_count =
      gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
          ->vehicle_types_.size();

  write_station_status(dir, 7U);
  run_update(c, gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  EXPECT_TRUE(first_provider_idx == p.idx_);
  EXPECT_EQ(first_vehicle_type_count, p.vehicle_types_.size());
  EXPECT_EQ(
      7U,
      p.stations_.at(std::string{kStation1}).status_.num_vehicles_available_);
  EXPECT_TRUE(p.has_vehicles_to_rent_);
}

TEST(motis, gbfs_update_updates_free_floating_vehicles_on_second_run) {
  auto const dir = make_temp_dir("vehicle-update");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);
  ASSERT_EQ("vehicle-1", gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                             ->vehicle_status_.front()
                             .id_);

  write_free_bike_status(dir, "vehicle-2", 49.8755, 8.6280);
  run_update(c, gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(1U, p.vehicle_status_.size());
  EXPECT_EQ("vehicle-2", p.vehicle_status_.front().id_);
  EXPECT_DOUBLE_EQ(49.8755, p.vehicle_status_.front().pos_.lat_);
  EXPECT_DOUBLE_EQ(8.6280, p.vehicle_status_.front().pos_.lng_);
  EXPECT_TRUE(p.has_vehicles_to_rent_);
}

TEST(motis, gbfs_update_ignores_timestamp_only_changes_for_routing_cache) {
  auto const dir = make_temp_dir("timestamp-hash");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);
  auto const provider_idx =
      gbfs->providers_.at(gbfs->provider_by_id_.at("test"))->idx_;
  auto const first_cached = gbfs->cache_.get(provider_idx);
  ASSERT_NE(nullptr, first_cached);

  write_station_status(dir, 3U, false, kBikeType, 41U);
  run_update(c, gbfs);
  auto const second_cached = gbfs->cache_.get(provider_idx);
  ASSERT_NE(nullptr, second_cached);
  EXPECT_EQ(first_cached.get(), second_cached.get());

  write_station_status(dir, 8U, false, kBikeType, 42U);
  run_update(c, gbfs);
  auto const third_cached = gbfs->cache_.get(provider_idx);
  ASSERT_NE(nullptr, third_cached);
  EXPECT_NE(first_cached.get(), third_cached.get());
  EXPECT_EQ(8U, gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                    ->stations_.at(std::string{kStation1})
                    .status_.num_vehicles_available_);
}

TEST(motis, gbfs_update_ttl_controls_refresh_behavior) {
  {
    auto const dir = make_temp_dir("ttl-3600");
    write_default_feed(dir);
    write_station_status(dir, 3U, false, kBikeType, 40U, 3600U);

    auto gbfs = std::shared_ptr<gbfs_data>{};
    auto const c = make_gbfs_config_without_ttl_overwrite(dir);
    run_update(c, gbfs);
    write_station_status(dir, 9U, false, kBikeType, 41U, 3600U);
    run_update(c, gbfs);

    EXPECT_EQ(3U, gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                      ->stations_.at(std::string{kStation1})
                      .status_.num_vehicles_available_);
  }

  {
    auto const dir = make_temp_dir("ttl-0");
    write_default_feed(dir);
    write_station_status(dir, 3U, false, kBikeType, 40U, 0U);

    auto gbfs = std::shared_ptr<gbfs_data>{};
    auto const c = make_gbfs_config_without_ttl_overwrite(dir);
    run_update(c, gbfs);
    write_station_status(dir, 9U, false, kBikeType, 41U, 0U);
    run_update(c, gbfs);

    EXPECT_EQ(9U, gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                      ->stations_.at(std::string{kStation1})
                      .status_.num_vehicles_available_);
  }
}

TEST(motis, gbfs_update_station_move_updates_provider_rtree) {
  auto const dir = make_temp_dir("station-rtree");
  write_discovery(dir, "");
  write_system_information(dir);
  write_vehicle_types(dir);
  write_station_information(dir);
  write_station_status(dir);
  write_empty_geofencing(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);

  auto const idx = gbfs->providers_.at(gbfs->provider_by_id_.at("test"))->idx_;
  auto const old_pos = geo::latlng{49.871651, 8.631084};
  auto const new_pos = geo::latlng{49.875309, 8.627667};
  EXPECT_TRUE(rtree_contains(*gbfs, old_pos, idx));
  EXPECT_FALSE(rtree_contains(*gbfs, new_pos, idx));

  write_station_information(dir, new_pos.lat_, new_pos.lng_);
  run_update(c, gbfs);

  EXPECT_FALSE(rtree_contains(*gbfs, old_pos, idx));
  EXPECT_TRUE(rtree_contains(*gbfs, new_pos, idx));
}

TEST(motis, gbfs_update_retains_previous_provider_on_malformed_refresh) {
  auto const dir = make_temp_dir("bad-refresh");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);
  ASSERT_EQ(1U, gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                    ->vehicle_status_.size());

  write_file(dir / "station_information.json", R"({"data": {"stations": )");
  run_update(c, gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(1U, p.stations_.size());
  EXPECT_TRUE(p.stations_.contains(std::string{kStation1}));
  ASSERT_EQ(1U, p.vehicle_status_.size());
  EXPECT_EQ("vehicle-1", p.vehicle_status_.front().id_);
}

TEST(motis, gbfs_update_discovery_removes_stale_vehicle_feed_data) {
  auto const dir = make_temp_dir("discovery-removes-vehicles");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);
  ASSERT_EQ(1U, gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                    ->vehicle_status_.size());

  write_discovery(dir, "");
  run_update(c, gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  EXPECT_TRUE(p.vehicle_status_.empty());
  EXPECT_EQ(0U, count_additional_vehicle_nodes(routing_data_for(*gbfs, p)));
}

TEST(motis, gbfs_update_recomputes_dependents_when_vehicle_types_change) {
  auto const dir = make_temp_dir("vehicle-types-update");
  write_default_feed(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir);
  run_update(c, gbfs);
  EXPECT_EQ(1U, gbfs->providers_.at(gbfs->provider_by_id_.at("test"))
                    ->vehicle_types_.size());

  write_vehicle_types(dir, true);
  write_station_status(dir, 4U, false, kScooterType);
  write_free_bike_status(dir, "scooter-1", 49.875309, 8.627667, false, false,
                         kScooterType);
  run_update(c, gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(2U, p.vehicle_types_.size());
  ASSERT_EQ(2U, p.products_.size());
  ASSERT_EQ(1U, p.vehicle_status_.size());
  auto const scooter_idx = p.vehicle_status_.front().vehicle_type_idx_;
  ASSERT_NE(vehicle_type_idx_t::invalid(), scooter_idx);
  EXPECT_EQ("scooter", p.vehicle_types_.at(scooter_idx).id_);
}

TEST(motis, gbfs_update_handles_inconsistent_references) {
  auto const dir = make_temp_dir("inconsistent");
  write_discovery(dir, "vehicle_status");
  write_system_information(dir);
  write_vehicle_types(dir);
  write_station_information(dir);
  write_station_status(dir, 5U, true, "missing-type");
  write_vehicle_status_at_station(dir, "missing-station", kBikeType);
  write_empty_geofencing(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(1U, p.stations_.size());
  EXPECT_EQ(
      5U,
      p.stations_.at(std::string{kStation1}).status_.num_vehicles_available_);
  EXPECT_TRUE(p.vehicle_status_.empty());
  EXPECT_GE(p.skipped_station_status_, 1U);
  EXPECT_GE(p.skipped_vehicle_status_, 1U);
}

TEST(motis, gbfs_update_uses_station_coordinates_for_docked_vehicle_status) {
  auto const dir = make_temp_dir("docked-vehicle");
  write_discovery(dir, "vehicle_status");
  write_system_information(dir);
  write_vehicle_types(dir);
  write_station_information(dir);
  write_station_status(dir, 1U);
  write_vehicle_status_at_station(dir, kStation1, kBikeType);
  write_empty_geofencing(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(1U, p.vehicle_status_.size());
  EXPECT_DOUBLE_EQ(p.stations_.at(std::string{kStation1}).info_.pos_.lat_,
                   p.vehicle_status_.front().pos_.lat_);
  EXPECT_DOUBLE_EQ(p.stations_.at(std::string{kStation1}).info_.pos_.lng_,
                   p.vehicle_status_.front().pos_.lng_);
}

TEST(motis, gbfs_update_geofence_can_suppress_free_floating_start_node) {
  auto const dir = make_temp_dir("geofence-no-start");
  write_default_feed(dir);
  write_geofencing_no_start_zone(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(1U, p.vehicle_status_.size());
  EXPECT_EQ(0U, count_additional_vehicle_nodes(routing_data_for(*gbfs, p)));
}

TEST(motis, gbfs_update_global_geofencing_rules) {
  auto const dir = make_temp_dir("global-geofence");
  write_default_feed(dir);
  write_geofencing_global_no_return(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  auto const prod_rd = gbfs->get_products_routing_data(
      *street_data().w_, *street_data().l_,
      gbfs_products_ref{p.idx_, gbfs_products_idx_t{0}});
  auto const n = osr::node_idx_t{0U};
  EXPECT_FALSE(prod_rd->end_allowed_.test(n));
  EXPECT_TRUE(prod_rd->through_allowed_.test(n));
}

TEST(motis, gbfs_update_station_area_marks_base_nodes_return_allowed) {
  auto const dir = make_temp_dir("station-area");
  write_discovery(dir, "");
  write_system_information(dir);
  write_vehicle_types(dir, false, "any_station");
  write_station_information_with_area(dir);
  write_station_status(dir, 3U);
  write_empty_geofencing(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  auto const prod_rd = gbfs->get_products_routing_data(
      *street_data().w_, *street_data().l_,
      gbfs_products_ref{p.idx_, gbfs_products_idx_t{0}});
  auto has_base_return_node = false;
  for (auto i = 0U; i != street_data().w_->n_nodes(); ++i) {
    has_base_return_node =
        has_base_return_node || prod_rd->end_allowed_.test(osr::node_idx_t{i});
  }
  EXPECT_TRUE(has_base_return_node);
}

TEST(motis, gbfs_update_supports_local_gbfs_manifest_provider_dirs) {
  auto const dir = make_temp_dir("manifest");
  auto const provider_dir = dir / "provider-a";
  write_default_feed(provider_dir);
  write_file(dir / "gbfs.json", R"({
    "last_updated": 1,
    "ttl": 0,
    "version": "3.0",
    "data": {
      "datasets": [
        {
          "system_id": "provider-a",
          "versions": [
            {"version": "2.3", "url": "provider-a"}
          ]
        },
        "bad",
        {"system_id": "missing-url", "versions": [{}]}
      ]
    }
  })");
  write_file(dir / "manifest.json", R"({
    "last_updated": 1,
    "ttl": 0,
    "version": "3.0",
    "data": {
      "datasets": [
        {
          "system_id": "provider-a",
          "versions": [
            {"version": "2.3", "url": "provider-a"}
          ]
        }
      ]
    }
  })");

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir, "agg"), gbfs);

  ASSERT_EQ(1U, gbfs->aggregated_feeds_->size());
  ASSERT_EQ(1U, gbfs->providers_.size());
  ASSERT_TRUE(gbfs->provider_by_id_.contains("agg:provider-a"));
  auto const& p =
      *gbfs->providers_.at(gbfs->provider_by_id_.at("agg:provider-a"));
  EXPECT_EQ("agg:provider-a", p.id_);
  ASSERT_EQ(1U, p.stations_.size());
  EXPECT_EQ("Test Bikes", p.sys_info_.name_);
}

TEST(motis, gbfs_update_local_manifest_update_adds_and_removes_providers) {
  auto const dir = make_temp_dir("manifest-update");
  write_default_feed(dir / "provider-a");
  write_default_feed(dir / "provider-b");
  write_system_information(dir / "provider-b", "Provider B", "#445566");
  write_manifest(dir, {"provider-a"});

  auto gbfs = std::shared_ptr<gbfs_data>{};
  auto const c = make_gbfs_config(dir, "agg");
  run_update(c, gbfs);

  auto const provider_a_idx = gbfs->provider_by_id_.at("agg:provider-a");
  ASSERT_NE(nullptr, gbfs->providers_.at(provider_a_idx));
  EXPECT_EQ(
      "agg:provider-a",
      gbfs->providers_.at(gbfs->provider_by_id_.at("agg:provider-a"))->id_);

  write_manifest(dir, {"provider-b"});
  run_update(c, gbfs);

  ASSERT_TRUE(gbfs->provider_by_id_.contains("agg:provider-a"));
  ASSERT_TRUE(gbfs->provider_by_id_.contains("agg:provider-b"));
  EXPECT_EQ(nullptr, gbfs->providers_.at(provider_a_idx));
  auto const provider_b_idx = gbfs->provider_by_id_.at("agg:provider-b");
  ASSERT_NE(nullptr, gbfs->providers_.at(provider_b_idx));
  EXPECT_FALSE(provider_a_idx == provider_b_idx);
  EXPECT_EQ("Provider B",
            gbfs->providers_.at(gbfs->provider_by_id_.at("agg:provider-b"))
                ->sys_info_.name_);
}

TEST(motis, gbfs_update_supports_lamassu_systems_manifest) {
  auto const dir = make_temp_dir("lamassu");
  write_default_feed(dir / "system-a");

  // Lamassu 2.3 aggregation format using a top-level "systems" array, mixed
  // with invalid entries that must be skipped
  auto const manifest = json::serialize(json::value{
      {"last_updated", 1},
      {"ttl", 0},
      {"version", "2.3"},
      {"systems",
       json::array{json::value{{"id", "system-a"}, {"url", "system-a"}},
                   json::value{{"id", ""}, {"url", "bad"}}, "not-an-object"}},
  });
  write_file(dir / "gbfs.json", manifest);
  write_file(dir / "manifest.json", manifest);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir, "agg"), gbfs);

  ASSERT_EQ(1U, gbfs->aggregated_feeds_->size());
  ASSERT_EQ(1U, gbfs->providers_.size());
  ASSERT_TRUE(gbfs->provider_by_id_.contains("agg:system-a"));
  auto const& p =
      *gbfs->providers_.at(gbfs->provider_by_id_.at("agg:system-a"));
  EXPECT_EQ("agg:system-a", p.id_);
  ASSERT_EQ(1U, p.stations_.size());
  EXPECT_EQ("Test Bikes", p.sys_info_.name_);
}

TEST(motis, gbfs_update_geofencing_unknown_type_ref) {
  auto const dir = make_temp_dir("unknown-type-geofence");
  write_default_feed(dir);
  write_geofencing_unknown_type_ref(dir);

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  EXPECT_TRUE(p.has_vehicles_to_rent_);
  EXPECT_FALSE(p.products_.empty());
}

TEST(motis, gbfs_update_handles_v1_feed_without_vehicle_types) {
  // GBFS 1.x style feed: no vehicle_types feed, count-only station status and
  // untyped free-floating bikes
  auto const dir = make_temp_dir("v1-feed");
  write_discovery(dir, "free_bike_status", /*include_geofencing=*/false,
                  /*include_vehicle_types=*/false);
  write_system_information(dir);
  write_station_information(dir);
  write_file(dir / "station_status.json", json::serialize(json::value{
                                              {"last_updated", 40},
                                              {"ttl", 0},
                                              {"version", "1.0"},
                                              {"data",
                                               {{"stations",
                                                 {{{"station_id", "station-1"},
                                                   {"num_bikes_available", 3},
                                                   {"is_renting", true},
                                                   {"is_returning", true}}}}}},
                                          }));
  // untyped bike (no vehicle_type_id) -> synthesized default bicycle type
  write_free_bike_status(dir, "bike-1", 49.875309, 8.627667, false, false, "");

  auto gbfs = std::shared_ptr<gbfs_data>{};
  run_update(make_gbfs_config(dir), gbfs);

  ASSERT_EQ(1U, gbfs->providers_.size());
  auto const& p = *gbfs->providers_.at(gbfs->provider_by_id_.at("test"));
  ASSERT_EQ(1U, p.stations_.size());
  EXPECT_EQ(
      3U,
      p.stations_.at(std::string{kStation1}).status_.num_vehicles_available_);
  ASSERT_EQ(1U, p.vehicle_status_.size());
  EXPECT_EQ("bike-1", p.vehicle_status_.front().id_);
  EXPECT_FALSE(p.products_.empty());
  EXPECT_TRUE(p.has_vehicles_to_rent_);
  EXPECT_GT(additional_node_count(*gbfs, p), 0U);
}
