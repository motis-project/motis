#include "gtest/gtest.h"

#include <chrono>
#include <optional>
#include <string>

#include "boost/json.hpp"

#include "motis/gbfs/parser.h"

namespace json = boost::json;

using namespace motis::gbfs;

TEST(motis, gbfs_parser_parse_discovery_v3_no_language_map) {
  auto const urls = parse_discovery(json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "version": "3.0",
    "data": {
      "feeds": [
        {"name": "system_information", "url": "https://example.com/v3/system"},
        {"name": "free_bike_status", "url": "https://example.com/v3/free_bikes"},
        {"name": "station_information", "url": "https://example.com/v3/stations"}
      ]
    }
  })"));

  ASSERT_EQ(3U, urls.size());
  EXPECT_EQ("https://example.com/v3/system", urls.at("system_information"));
  EXPECT_EQ("https://example.com/v3/free_bikes", urls.at("free_bike_status"));
  EXPECT_EQ("https://example.com/v3/stations", urls.at("station_information"));
}

TEST(motis, gbfs_parser_parse_discovery_v2_language_map) {
  auto const urls = parse_discovery(json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "data": {
      "en": {
        "feeds": [
          {"name": "system_information", "url": "https://example.com/system"},
          {"name": "free_bike_status", "url": "https://example.com/free_bikes"}
        ]
      }
    }
  })"));

  ASSERT_EQ(2U, urls.size());
  EXPECT_EQ("https://example.com/system", urls.at("system_information"));
  EXPECT_EQ("https://example.com/free_bikes", urls.at("free_bike_status"));
}

TEST(motis, gbfs_parser_loads_v2_station_feed) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_system_information(provider, json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "version": "2.3",
    "data": {
      "system_id": "sys",
      "name": "Test Provider",
      "brand_assets": {"color": "#123456"}
    }
  })"));

  load_vehicle_types(provider, json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "version": "2.3",
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "bike",
          "form_factor": "bicycle",
          "propulsion_type": "human"
        }
      ]
    }
  })"));

  load_station_information(provider, json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "version": "2.3",
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "name": "Main Station",
          "lat": 48.1,
          "lon": 11.5
        }
      ]
    }
  })"));

  load_station_status(provider, json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "version": "2.3",
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "num_bikes_available": 3,
          "is_renting": true,
          "is_returning": true,
          "vehicle_types_available": [
            {"vehicle_type_id": "bike", "count": 3}
          ]
        }
      ]
    }
  })"));

  EXPECT_EQ("sys", provider.sys_info_.id_);
  EXPECT_EQ("Test Provider", provider.sys_info_.name_);
  EXPECT_EQ("#123456", provider.sys_info_.color_);
  ASSERT_EQ(1U, provider.vehicle_types_.size());
  EXPECT_EQ(vehicle_form_factor::kBicycle,
            provider.vehicle_types_.front().form_factor_);
  ASSERT_EQ(1U, provider.stations_.size());

  auto const& station = provider.stations_.at("station-1");
  EXPECT_EQ("Main Station", station.info_.name_);
  EXPECT_DOUBLE_EQ(48.1, station.info_.pos_.lat_);
  EXPECT_DOUBLE_EQ(11.5, station.info_.pos_.lng_);
  EXPECT_EQ(3U, station.status_.num_vehicles_available_);
  EXPECT_TRUE(station.status_.is_renting_);
  EXPECT_TRUE(station.status_.is_returning_);
}

TEST(motis, gbfs_parser_prefers_v3_vehicle_status_fields) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "version": "3.0",
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "scooter",
          "form_factor": "scooter_standing",
          "propulsion_type": "electric",
          "return_constraint": "free_floating"
        }
      ]
    }
  })"));

  load_vehicle_status(provider, json::parse(R"({
    "version": "2.3",
    "data": {
      "vehicles": [
        {
          "vehicle_id": "vehicle-v3",
          "bike_id": "vehicle-v2",
          "lat": 48.3,
          "lon": 11.4,
          "vehicle_type_id": "scooter",
          "is_reserved": false,
          "is_disabled": false
        }
      ],
      "bikes": [
        {
          "bike_id": "bike-v2",
          "lat": 1.0,
          "lon": 2.0
        }
      ]
    }
  })"));

  ASSERT_EQ(1U, provider.vehicle_status_.size());
  EXPECT_EQ("vehicle-v3", provider.vehicle_status_.front().id_);
  EXPECT_DOUBLE_EQ(48.3, provider.vehicle_status_.front().pos_.lat_);
  EXPECT_DOUBLE_EQ(11.4, provider.vehicle_status_.front().pos_.lng_);
}

TEST(motis, gbfs_parser_skips_bad_entries_and_accepts_lenient_values) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_station_information(provider, json::parse(R"({
    "version": "2.3",
    "data": {
      "stations": [
        "invalid",
        {"station_id": "missing-position"},
        {
          "station_id": 42,
          "name": [{"language": "en"}, {"language": "de", "text": 123}],
          "lat": "48.5",
          "lon": "11.6"
        }
      ]
    }
  })"));

  ASSERT_EQ(1U, provider.stations_.size());
  auto const& station = provider.stations_.at("42");
  EXPECT_EQ("123", station.info_.name_);
  EXPECT_DOUBLE_EQ(48.5, station.info_.pos_.lat_);
  EXPECT_DOUBLE_EQ(11.6, station.info_.pos_.lng_);
  EXPECT_EQ(2U, provider.skipped_station_infos_);
}

TEST(motis, gbfs_parser_accepts_lenient_status_counts_and_booleans) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_station_information(provider, json::parse(R"({
    "data": {
      "stations": [
        {"station_id": "station-1", "name": "Station", "lat": 1.0, "lon": 2.0}
      ]
    }
  })"));

  load_station_status(provider, json::parse(R"({
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "num_vehicles_available": "4",
          "is_renting": "yes",
          "is_returning": "0"
        }
      ]
    }
  })"));

  auto const& status = provider.stations_.at("station-1").status_;
  EXPECT_EQ(4U, status.num_vehicles_available_);
  EXPECT_TRUE(status.is_renting_);
  EXPECT_FALSE(status.is_returning_);
}

TEST(motis, gbfs_parser_ignores_unknown_station_status_vehicle_types) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "bike",
          "form_factor": "bicycle",
          "propulsion_type": "human",
          "return_constraint": "any_station"
        }
      ]
    }
  })"));

  load_station_information(provider, json::parse(R"({
    "data": {
      "stations": [
        {"station_id": "station-1", "name": "Station", "lat": 1.0, "lon": 2.0}
      ]
    }
  })"));

  load_station_status(provider, json::parse(R"({
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "num_vehicles_available": 14,
          "vehicle_types_available": [
            {"vehicle_type_id": "unknown", "count": 9},
            {"vehicle_type_id": "bike", "count": 5}
          ]
        }
      ]
    }
  })"));

  auto const& status = provider.stations_.at("station-1").status_;
  EXPECT_EQ(5U, status.num_vehicles_available_);
  ASSERT_EQ(1U, status.vehicle_types_available_.size());
  EXPECT_EQ(5U, status.vehicle_types_available_.begin()->second);
  EXPECT_EQ(1U, provider.skipped_station_status_);
}

TEST(motis, gbfs_parser_infers_return_constraint_by_start_type) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "bike",
          "form_factor": "bicycle",
          "propulsion_type": "human"
        }
      ]
    }
  })"));

  load_station_information(provider, json::parse(R"({
    "data": {
      "stations": [
        {"station_id": "station-1", "name": "Station", "lat": 1.0, "lon": 2.0}
      ]
    }
  })"));

  load_station_status(provider, json::parse(R"({
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "vehicle_types_available": [
            {"vehicle_type_id": "bike", "count": 2}
          ]
        }
      ]
    }
  })"));

  load_vehicle_status(provider, json::parse(R"({
    "data": {
      "vehicles": [
        {
          "vehicle_id": "vehicle-1",
          "lat": 1.1,
          "lon": 2.1,
          "vehicle_type_id": "bike"
        }
      ]
    }
  })"));

  ASSERT_EQ(2U, provider.vehicle_types_.size());
  auto const station_type = provider.stations_.at("station-1")
                                .status_.vehicle_types_available_.begin()
                                ->first;
  auto const free_floating_type =
      provider.vehicle_status_.front().vehicle_type_idx_;
  EXPECT_EQ(return_constraint::kAnyStation,
            provider.vehicle_types_.at(station_type).return_constraint_);
  EXPECT_EQ(return_constraint::kFreeFloating,
            provider.vehicle_types_.at(free_floating_type).return_constraint_);
}

TEST(motis, gbfs_parser_vehicle_status_uses_station_position_when_missing) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "bike",
          "form_factor": "bicycle",
          "propulsion_type": "human",
          "return_constraint": "any_station"
        }
      ]
    }
  })"));

  load_station_information(provider, json::parse(R"({
    "data": {
      "stations": [
        {"station_id": "station-1", "name": "Station", "lat": 48.1, "lon": 11.2}
      ]
    }
  })"));

  load_vehicle_status(provider, json::parse(R"({
    "data": {
      "vehicles": [
        {
          "vehicle_id": "vehicle-1",
          "station_id": "station-1",
          "vehicle_type_id": "bike"
        },
        {
          "vehicle_id": "vehicle-2",
          "station_id": "missing",
          "vehicle_type_id": "bike"
        }
      ]
    }
  })"));

  ASSERT_EQ(1U, provider.vehicle_status_.size());
  EXPECT_DOUBLE_EQ(48.1, provider.vehicle_status_.front().pos_.lat_);
  EXPECT_DOUBLE_EQ(11.2, provider.vehicle_status_.front().pos_.lng_);
  EXPECT_EQ(1U, provider.skipped_vehicle_status_);
}

TEST(motis, gbfs_parser_geofencing_accepts_v3_and_v2_rule_fields) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "bike",
          "form_factor": "bicycle",
          "propulsion_type": "human",
          "return_constraint": "free_floating"
        }
      ]
    }
  })"));

  load_geofencing_zones(provider, json::parse(R"({
    "data": {
      "geofencing_zones": {
        "type": "FeatureCollection",
        "features": [
          {
            "type": "Feature",
            "properties": {
              "name": [{"text": "Zone"}],
              "rules": [
                {
                  "vehicle_type_ids": ["bike"],
                  "ride_allowed": false,
                  "ride_start_allowed": true
                },
                {
                  "vehicle_type_id": "bike",
                  "ride_allowed": true
                }
              ]
            },
            "geometry": {
              "type": "MultiPolygon",
              "coordinates": [[[
                [11.0, 48.0],
                [11.1, 48.0],
                [11.1, 48.1],
                [11.0, 48.0]
              ]]]
            }
          }
        ]
      },
      "global_rules": [
        {
          "vehicle_type_id": "bike",
          "ride_allowed": false,
          "ride_through_allowed": true
        }
      ]
    }
  })"));

  ASSERT_EQ(1U, provider.geofencing_zones_.zones_.size());
  auto const& rules = provider.geofencing_zones_.zones_.front().rules_;
  ASSERT_EQ(2U, rules.size());
  EXPECT_TRUE(rules[0].ride_start_allowed_);
  EXPECT_FALSE(rules[0].ride_end_allowed_);
  EXPECT_FALSE(rules[0].ride_through_allowed_);
  EXPECT_TRUE(rules[1].ride_start_allowed_);
  EXPECT_TRUE(rules[1].ride_end_allowed_);
  EXPECT_TRUE(rules[1].ride_through_allowed_);

  ASSERT_EQ(1U, provider.geofencing_zones_.global_rules_.size());
  EXPECT_FALSE(
      provider.geofencing_zones_.global_rules_.front().ride_start_allowed_);
  EXPECT_FALSE(
      provider.geofencing_zones_.global_rules_.front().ride_end_allowed_);
  EXPECT_TRUE(
      provider.geofencing_zones_.global_rules_.front().ride_through_allowed_);
}

TEST(motis, gbfs_parser_geofencing_skips_bad_parts_and_unknown_type_rules) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "data": {
      "vehicle_types": [
        {
          "vehicle_type_id": "bike",
          "form_factor": "bicycle",
          "propulsion_type": "human",
          "return_constraint": "free_floating"
        }
      ]
    }
  })"));

  load_geofencing_zones(provider, json::parse(R"({
    "data": {
      "geofencing_zones": {
        "type": "FeatureCollection",
        "features": [
          "bad",
          {
            "type": "Feature",
            "properties": {"name": "No Rules"},
            "geometry": {"type": "MultiPolygon", "coordinates": []}
          },
          {
            "type": "Feature",
            "properties": {
              "name": "Valid",
              "rules": [
                "bad rule",
                {"vehicle_type_ids": ["unknown"], "ride_allowed": false},
                {"vehicle_type_ids": ["bike"], "station_parking": true}
              ]
            },
            "geometry": {
              "type": "MultiPolygon",
              "coordinates": [[[
                [11.0, 48.0],
                [11.1, 48.0],
                [11.1, 48.1],
                [11.0, 48.0]
              ]]]
            }
          }
        ]
      }
    }
  })"));

  ASSERT_EQ(1U, provider.geofencing_zones_.zones_.size());
  auto const& rules = provider.geofencing_zones_.zones_.front().rules_;
  ASSERT_EQ(2U, rules.size());
  ASSERT_EQ(1U, rules[0].vehicle_type_idxs_.size());
  EXPECT_TRUE(vehicle_type_idx_t::invalid() ==
              rules[0].vehicle_type_idxs_.front());
  ASSERT_TRUE(rules[1].station_parking_.has_value());
  EXPECT_TRUE(*rules[1].station_parking_);
  EXPECT_EQ(2U, provider.skipped_geofencing_zones_);
  EXPECT_EQ(1U, provider.skipped_geofencing_rules_);
}

TEST(motis, gbfs_parser_parse_timestamp_supports_multiple_formats) {
  auto const seconds =
      [](std::optional<std::chrono::system_clock::time_point> const& tp) {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   tp->time_since_epoch())
            .count();
      };

  // POSIX timestamp as integer (GBFS 1.x / 2.x)
  auto const as_int = parse_timestamp(json::value{1700000000});
  ASSERT_TRUE(as_int.has_value());
  EXPECT_EQ(1700000000, seconds(as_int));

  // POSIX timestamp as numeric string
  auto const as_str = parse_timestamp(json::value{"1700000000"});
  ASSERT_TRUE(as_str.has_value());
  EXPECT_EQ(1700000000, seconds(as_str));

  // RFC3339 / ISO 8601 timestamps (GBFS 3.x)
  auto const utc = parse_timestamp(json::value{"2020-01-01T00:00:00Z"});
  auto const naive = parse_timestamp(json::value{"2020-01-01T00:00:00"});
  auto const offset = parse_timestamp(json::value{"2020-01-01T01:00:00+01:00"});
  ASSERT_TRUE(utc.has_value());
  ASSERT_TRUE(naive.has_value());
  ASSERT_TRUE(offset.has_value());
  EXPECT_TRUE(utc == naive);
  EXPECT_TRUE(utc == offset);

  // unparseable values
  EXPECT_FALSE(parse_timestamp(json::value{"not-a-timestamp"}).has_value());
  EXPECT_FALSE(parse_timestamp(json::value{}).has_value());
  EXPECT_FALSE(parse_timestamp(json::value{true}).has_value());
}

TEST(motis, gbfs_parser_discovery_is_lenient_with_bad_feed_entries) {
  auto const urls = parse_discovery(json::parse(R"({
    "last_updated": 1,
    "ttl": 60,
    "version": "3.0",
    "data": {
      "feeds": [
        "not-an-object",
        {"name": "system_information"},
        {"url": "https://example.com/no-name"},
        {"name": "station_information", "url": "https://example.com/stations"}
      ]
    }
  })"));

  ASSERT_EQ(1U, urls.size());
  EXPECT_EQ("https://example.com/stations", urls.at("station_information"));

  EXPECT_TRUE(parse_discovery(json::parse(R"({"data": {}})")).empty());
}

TEST(motis, gbfs_parser_v1_free_bike_status_without_vehicle_types) {
  // GBFS 1.x feeds have no vehicle_types and only free_bike_status with bikes
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_status(provider, json::parse(R"({
    "last_updated": 1,
    "ttl": 0,
    "data": {
      "bikes": [
        {"bike_id": "bike-1", "lat": 48.1, "lon": 11.5},
        {"bike_id": "bike-2", "lat": 48.2, "lon": 11.6}
      ]
    }
  })"));

  ASSERT_EQ(2U, provider.vehicle_status_.size());
  // a default bicycle vehicle type is created for untyped vehicles
  auto const type_idx = provider.vehicle_status_.front().vehicle_type_idx_;
  ASSERT_NE(vehicle_type_idx_t::invalid(), type_idx);
  EXPECT_EQ(vehicle_form_factor::kBicycle,
            provider.vehicle_types_.at(type_idx).form_factor_);
  EXPECT_EQ(return_constraint::kFreeFloating,
            provider.vehicle_types_.at(type_idx).return_constraint_);
}

TEST(motis, gbfs_parser_vehicle_types_cover_form_factors_and_propulsion) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_vehicle_types(provider, json::parse(R"({
    "data": {
      "vehicle_types": [
        {"vehicle_type_id": "a", "form_factor": "bike",
         "propulsion_type": "electric_assist", "return_constraint": "any_station"},
        {"vehicle_type_id": "b", "form_factor": "cargo_bicycle",
         "propulsion_type": "hybrid", "return_constraint": "roundtrip_station"},
        {"vehicle_type_id": "c", "form_factor": "car",
         "propulsion_type": "combustion", "return_constraint": "free_floating"},
        {"vehicle_type_id": "d", "form_factor": "moped",
         "propulsion_type": "electric", "return_constraint": "hybrid"},
        {"vehicle_type_id": "e", "form_factor": "scooter",
         "propulsion_type": "plug_in_hybrid", "return_constraint": "any_station"},
        {"vehicle_type_id": "f", "form_factor": "spaceship",
         "propulsion_type": "warp_drive", "return_constraint": "any_station"}
      ]
    }
  })"));

  ASSERT_EQ(6U, provider.vehicle_types_.size());
  auto const& vts = provider.vehicle_types_;
  auto const vt = [&](unsigned const i) -> vehicle_type const& {
    return vts.at(vehicle_type_idx_t{i});
  };

  // non-standard "bike" form factor maps to bicycle
  EXPECT_EQ(vehicle_form_factor::kBicycle, vt(0).form_factor_);
  EXPECT_EQ(propulsion_type::kElectricAssist, vt(0).propulsion_type_);
  EXPECT_EQ(return_constraint::kAnyStation, vt(0).return_constraint_);

  EXPECT_EQ(vehicle_form_factor::kCargoBicycle, vt(1).form_factor_);
  EXPECT_EQ(propulsion_type::kHybrid, vt(1).propulsion_type_);
  EXPECT_EQ(return_constraint::kRoundtripStation, vt(1).return_constraint_);

  EXPECT_EQ(vehicle_form_factor::kCar, vt(2).form_factor_);
  EXPECT_EQ(propulsion_type::kCombustion, vt(2).propulsion_type_);
  EXPECT_EQ(return_constraint::kFreeFloating, vt(2).return_constraint_);

  EXPECT_EQ(vehicle_form_factor::kMoped, vt(3).form_factor_);
  EXPECT_EQ(propulsion_type::kElectric, vt(3).propulsion_type_);
  // "hybrid" return constraint maps to free floating
  EXPECT_EQ(return_constraint::kFreeFloating, vt(3).return_constraint_);

  // pre-3.0 "scooter" form factor maps to standing scooter
  EXPECT_EQ(vehicle_form_factor::kScooterStanding, vt(4).form_factor_);
  EXPECT_EQ(propulsion_type::kPlugInHybrid, vt(4).propulsion_type_);

  // unknown form factor / propulsion type fall back to defaults
  EXPECT_EQ(vehicle_form_factor::kOther, vt(5).form_factor_);
  EXPECT_EQ(propulsion_type::kHuman, vt(5).propulsion_type_);

  // all of these have an explicit return constraint
  EXPECT_TRUE(vt(0).known_return_constraint_);
  EXPECT_TRUE(vt(1).known_return_constraint_);
  EXPECT_TRUE(vt(2).known_return_constraint_);
  EXPECT_TRUE(vt(3).known_return_constraint_);
  EXPECT_TRUE(vt(4).known_return_constraint_);
  EXPECT_TRUE(vt(5).known_return_constraint_);
}

TEST(motis, gbfs_parser_station_status_count_only_creates_default_type) {
  auto provider = gbfs_provider{.id_ = "provider"};

  load_station_information(provider, json::parse(R"({
    "data": {
      "stations": [
        {"station_id": "station-1", "name": "Station", "lat": 1.0, "lon": 2.0}
      ]
    }
  })"));

  load_station_status(provider, json::parse(R"({
    "data": {
      "stations": [
        {"station_id": "station-1", "num_bikes_available": 5}
      ]
    }
  })"));

  auto const& status = provider.stations_.at("station-1").status_;
  EXPECT_EQ(5U, status.num_vehicles_available_);
  ASSERT_EQ(1U, status.vehicle_types_available_.size());
  EXPECT_EQ(5U, status.vehicle_types_available_.begin()->second);
  // a default vehicle type was created for the untyped count
  ASSERT_EQ(1U, provider.vehicle_types_.size());
  EXPECT_EQ(vehicle_form_factor::kBicycle,
            provider.vehicle_types_.front().form_factor_);
}

TEST(motis, gbfs_parser_system_information_localized_name_and_defaults) {
  auto provider = gbfs_provider{.id_ = "fallback-id"};

  load_system_information(provider, json::parse(R"({
    "data": {
      "name": [{"language": "en", "text": "Localized Bikes"}],
      "name_short": [{"text": "LB"}]
    }
  })"));

  // missing system_id falls back to the provider id
  EXPECT_EQ("fallback-id", provider.sys_info_.id_);
  EXPECT_EQ("Localized Bikes", provider.sys_info_.name_);
  EXPECT_EQ("LB", provider.sys_info_.name_short_);
  // no brand_assets -> empty color
  EXPECT_TRUE(provider.sys_info_.color_.empty());
}

TEST(motis, gbfs_parser_invalid_station_area_is_ignored) {
  auto provider = gbfs_provider{.id_ = "provider"};

  // station_area has a degenerate ring (< 3 points) -> parsing fails, but the
  // station itself must still be loaded
  load_station_information(provider, json::parse(R"({
    "data": {
      "stations": [
        {
          "station_id": "station-1",
          "name": "Station",
          "lat": 1.0,
          "lon": 2.0,
          "station_area": {
            "type": "MultiPolygon",
            "coordinates": [[[[1.0, 1.0], [2.0, 2.0]]]]
          }
        }
      ]
    }
  })"));

  ASSERT_EQ(1U, provider.stations_.size());
  auto const& station = provider.stations_.at("station-1");
  EXPECT_DOUBLE_EQ(1.0, station.info_.pos_.lat_);
  EXPECT_EQ(nullptr, station.info_.station_area_.get());
}

TEST(motis, gbfs_parser_structurally_invalid_feeds_throw) {
  auto provider = gbfs_provider{.id_ = "provider"};

  // missing "data" object
  EXPECT_ANY_THROW(
      load_station_information(provider, json::parse(R"({"version": "2.3"})")));

  // vehicle status without a vehicles/bikes array
  EXPECT_ANY_THROW(load_vehicle_status(provider, json::parse(R"({
    "data": {"foo": 1}
  })")));

  // geofencing zones that are not a FeatureCollection
  EXPECT_ANY_THROW(load_geofencing_zones(provider, json::parse(R"({
    "data": {"geofencing_zones": "not-an-object"}
  })")));
}
