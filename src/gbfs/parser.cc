#include <string_view>

#include "motis/gbfs/parser.h"

#include "cista/hash.h"

namespace json = boost::json;

namespace motis::gbfs {

gbfs_version get_version(json::value const& root) {
  auto const& root_obj = root.as_object();
  if (!root_obj.contains("version")) {
    // 1.0 doesn't have the version key
    return gbfs_version::k1;
  }
  auto const version =
      static_cast<std::string_view>(root.at("version").as_string());
  if (version.starts_with("1.")) {
    return gbfs_version::k1;
  } else if (version.starts_with("2.")) {
    return gbfs_version::k2;
  } else if (version.starts_with("3.")) {
    return gbfs_version::k3;
  } else {
    throw utl::fail("unsupported GBFS version: {}", version);
  }
}

std::string get_localized_string(gbfs_version const version,
                                 json::value const& v) {
  if (version == gbfs_version::k3) {
    auto const& arr = v.as_array();
    if (!arr.empty()) {
      return static_cast<std::string>(
          arr[0].as_object().at("text").as_string());
    }
    return "";
  } else {
    return static_cast<std::string>(v.as_string());
  }
}

std::string optional_str(json::object const& obj, std::string_view key) {
  return obj.contains(key) ? static_cast<std::string>(obj.at(key).as_string())
                           : "";
}

bool get_bool(gbfs_version const version,
              json::object const& obj,
              std::string_view const key) {
  return version == gbfs_version::k1 ? obj.at(key).to_number<int>() == 1
                                     : obj.at(key).as_bool();
}

hash_map<std::string, std::string> parse_discovery(json::value const& root) {
  auto urls = hash_map<std::string, std::string>{};

  auto const& data = root.at("data").as_object();
  if (data.empty()) {
    return urls;
  }
  auto const& feeds =
      data.contains("feeds")
          ? data.at("feeds").as_array()
          : data.begin()->value().as_object().at("feeds").as_array();

  for (auto const& feed : feeds) {
    auto const& name =
        static_cast<std::string>(feed.as_object().at("name").as_string());
    auto const& url =
        static_cast<std::string>(feed.as_object().at("url").as_string());
    urls[name] = url;
  }
  return urls;
}

rental_uris parse_rental_uris(json::object const& parent) {
  auto uris = rental_uris{};

  if (parent.contains("rental_uris")) {
    auto const& o = parent.at("rental_uris").as_object();
    uris.android_ = optional_str(o, "android");
    uris.ios_ = optional_str(o, "ios");
    uris.web_ = optional_str(o, "web");
  }

  return uris;
}

void load_system_information(gbfs_provider& provider, json::value const& root) {
  auto const version = get_version(root);
  auto const& data = root.at("data").as_object();

  auto& si = provider.sys_info_;
  si.id_ = static_cast<std::string>(data.at("system_id").as_string());
  si.name_ = get_localized_string(version, data.at("name"));
  si.name_short_ = data.contains("short_name")
                       ? get_localized_string(version, data.at("short_name"))
                       : "";
  si.operator_ = data.contains("operator")
                     ? get_localized_string(version, data.at("operator"))
                     : "";
  si.url_ = optional_str(data, "url");
  si.purchase_url_ = optional_str(data, "purchase_url");
  si.mail_ = optional_str(data, "email");
}

void load_station_information(gbfs_provider& provider,
                              json::value const& root) {
  provider.stations_.clear();

  auto const version = get_version(root);

  auto const& stations_arr = root.at("data").at("stations").as_array();
  for (auto const& s : stations_arr) {
    auto const& station_obj = s.as_object();
    auto const station_id =
        static_cast<std::string>(station_obj.at("station_id").as_string());
    auto const name =
        version == gbfs_version::k2
            ? static_cast<std::string>(station_obj.at("name").as_string())
            : get_localized_string(version, station_obj.at("name"));
    auto const lat = station_obj.at("lat").as_double();
    auto const lon = station_obj.at("lon").as_double();

    tg_geom* area = nullptr;
    if (station_obj.contains("station_area")) {
      area = tg_parse_geojson(
          json::serialize(station_obj.at("station_area")).c_str());
      if (auto const* err = tg_geom_error(area); err != nullptr) {
        std::cerr << "[GBFS] failed to parse station_area geojson: " << err
                  << "\n";
        tg_geom_free(area);
        area = nullptr;
      }
      if (area != nullptr && tg_geom_typeof(area) != TG_MULTIPOLYGON) {
        std::cerr << "[GBFS] station_area must be a MultiPolygon, not "
                  << tg_geom_type_string(tg_geom_typeof(area)) << "\n";
        tg_geom_free(area);
        area = nullptr;
      }
    }

    provider.stations_[station_id] =
        station{.info_ = {.id_ = station_id,
                          .name_ = name,
                          .pos_ = geo::latlng{lat, lon},
                          .rental_uris_ = parse_rental_uris(station_obj),
                          .station_area_ =
                              std::unique_ptr<tg_geom, tg_geom_deleter>(area)}};
  }
}

void load_station_status(gbfs_provider& provider, json::value const& root) {
  auto const version = get_version(root);
  auto const& stations_arr = root.at("data").at("stations").as_array();
  for (auto const& s : stations_arr) {
    auto const& station_obj = s.as_object();
    auto const station_id =
        static_cast<std::string>(station_obj.at("station_id").as_string());
    auto const num_vehicles_available_key = version == gbfs_version::k2
                                                ? "num_bikes_available"
                                                : "num_vehicles_available";

    auto& station = provider.stations_.at(station_id);
    station.status_ = station_status{
        .num_vehicles_available_ =
            station_obj.at(num_vehicles_available_key).to_number<unsigned>(),
        .is_renting_ = get_bool(version, station_obj, "is_renting"),
        .is_returning_ = get_bool(version, station_obj, "is_returning")};
    if (station_obj.contains("vehicle_types_available")) {
      auto const& vta = station_obj.at("vehicle_types_available").as_array();
      auto unrestricted_available = 0U;
      auto any_station_available = 0U;
      for (auto const& vt : vta) {
        auto const vehicle_type_id =
            static_cast<std::string>(vt.at("vehicle_type_id").as_string());
        auto const count = vt.at("count").to_number<unsigned>();
        station.status_.vehicle_types_available_[vehicle_type_id] = count;
        if (auto const it = provider.vehicle_types_.find(vehicle_type_id);
            it != end(provider.vehicle_types_)) {
          switch (it->second.return_constraint_) {
            case return_constraint::kNone: ++unrestricted_available; break;
            case return_constraint::kAnyStation: ++any_station_available; break;
            case return_constraint::kRoundtripStation: break;
          }
        }
      }
      station.status_.num_vehicles_available_ =
          unrestricted_available + any_station_available;
    }
  }
}

vehicle_form_factor parse_form_factor(std::string_view const s) {
  switch (cista::hash(s)) {
    case cista::hash("bicycle"): return vehicle_form_factor::kBicycle;
    case cista::hash("cargo_bicycle"):
      return vehicle_form_factor::kCargoBicycle;
    case cista::hash("car"): return vehicle_form_factor::kCar;
    case cista::hash("moped"): return vehicle_form_factor::kMoped;
    case cista::hash("scooter_standing"):
      return vehicle_form_factor::kScooterStanding;
    case cista::hash("scooter_seated"):
      return vehicle_form_factor::kScooterSeated;
    case cista::hash("other"):
    default: return vehicle_form_factor::kOther;
  }
}

return_constraint parse_return_constraint(json::object const& vt) {
  if (vt.contains("return_constraint")) {
    switch (cista::hash(static_cast<std::string_view>(
        vt.at("return_constraint").as_string()))) {
      case cista::hash("any_station"): return return_constraint::kAnyStation;
      case cista::hash("roundtrip_station"):
        return return_constraint::kRoundtripStation;
      case cista::hash("free_floating"):
      case cista::hash("hybrid"):
      default: return return_constraint::kNone;
    }
  }
  return return_constraint::kNone;
}

void load_vehicle_types(gbfs_provider& provider, json::value const& root) {
  provider.vehicle_types_.clear();
  for (auto const& v : root.at("data").at("vehicle_types").as_array()) {
    auto const id =
        static_cast<std::string>(v.at("vehicle_type_id").as_string());
    provider.vehicle_types_[id] = vehicle_type{
        .id_ = id,
        .form_factor_ = parse_form_factor(
            static_cast<std::string_view>(v.at("form_factor").as_string())),
        .return_constraint_ = parse_return_constraint(v.as_object())};
  }
}

void load_vehicle_status(gbfs_provider& provider, json::value const& root) {
  provider.vehicle_status_.clear();

  auto const version = get_version(root);
  auto const& vehicles_arr =
      root.at("data")
          .at(version == gbfs_version::k3 ? "vehicles" : "bikes")
          .as_array();
  for (auto const& v : vehicles_arr) {
    auto const& vehicle_obj = v.as_object();

    if (!vehicle_obj.contains("lat") || !vehicle_obj.contains("lon") ||
        vehicle_obj.contains("station_id")) {
      // we only care about free-floating vehicles here
      continue;
    }

    auto const id = static_cast<std::string>(
        vehicle_obj.at(version == gbfs_version::k3 ? "vehicle_id" : "bike_id")
            .as_string());

    auto const type_id = optional_str(vehicle_obj, "vehicle_type_id");

    if (auto const it = provider.vehicle_types_.find(type_id);
        it != end(provider.vehicle_types_) &&
        it->second.return_constraint_ == return_constraint::kRoundtripStation) {
      // roundtrip vehicles currently not supported
      continue;
    }

    provider.vehicle_status_.emplace_back(vehicle_status{
        .id_ = id,
        .pos_ = geo::latlng{vehicle_obj.at("lat").as_double(),
                            vehicle_obj.at("lon").as_double()},
        .is_reserved_ = get_bool(version, vehicle_obj, "is_reserved"),
        .is_disabled_ = get_bool(version, vehicle_obj, "is_disabled"),
        .vehicle_type_id_ = type_id,
        .station_id_ = optional_str(vehicle_obj, "station_id"),
        .home_station_id_ = optional_str(vehicle_obj, "home_station_id"),
        .rental_uris_ = parse_rental_uris(vehicle_obj)});
  }
}

rule parse_rule(gbfs_version const version, json::value const& r) {
  auto const vti_key =
      version == gbfs_version::k2 ? "vehicle_type_id" : "vehicle_type_ids";
  auto const& rule_obj = r.as_object();
  return rule{
      .vehicle_type_ids_ =
          rule_obj.contains(vti_key)
              ? utl::to_vec(rule_obj.at(vti_key).as_array(),
                            [](auto const& vt) {
                              return static_cast<std::string>(vt.as_string());
                            })
              : std::vector<std::string>{},
      .ride_start_allowed_ = version == gbfs_version::k2
                                 ? rule_obj.at("ride_allowed").as_bool()
                                 : rule_obj.at("ride_start_allowed").as_bool(),
      .ride_end_allowed_ = version == gbfs_version::k2
                               ? rule_obj.at("ride_allowed").as_bool()
                               : rule_obj.at("ride_end_allowed").as_bool(),
      .ride_through_allowed_ = rule_obj.at("ride_through_allowed").as_bool(),
      .station_parking_ =
          rule_obj.contains("station_parking")
              ? std::optional{rule_obj.at("station_parking").as_bool()}
              : std::nullopt};
}

void load_geofencing_zones(gbfs_provider& provider, json::value const& root) {
  auto const version = get_version(root);

  auto const& zones_obj = root.at("data").at("geofencing_zones").as_object();
  utl::verify(zones_obj.at("type") == "FeatureCollection",
              "invalid geofencing_zones");

  auto zones =
      utl::to_vec(zones_obj.at("features").as_array(), [&](auto const& z) {
        auto const& props = z.at("properties").as_object();
        auto rules =
            utl::to_vec(props.at("rules").as_array(),
                        [&](auto const& r) { return parse_rule(version, r); });

        auto const feature_str = json::serialize(z);
        auto* geom = tg_parse_geojson(feature_str.c_str());
        if (auto const* err = tg_geom_error(geom); err != nullptr) {
          auto const msg = std::string{err};
          tg_geom_free(geom);
          throw utl::fail("failed to parse geojson: {}", msg);
        }
        if (tg_geom_typeof(geom) != TG_MULTIPOLYGON) {
          auto const type_str = tg_geom_type_string(tg_geom_typeof(geom));
          tg_geom_free(geom);
          throw utl::fail("geofencing zone must be a MultiPolygon, not {}",
                          type_str);
        }

        auto name = props.contains("name")
                        ? get_localized_string(version, props.at("name"))
                        : std::string{};

        return zone{geom, std::move(rules), std::move(name)};
      });

  //  required in 3.0, but some feeds don't have it
  auto global_rules =
      root.at("data").as_object().contains("global_rules")
          ? utl::to_vec(root.at("data").at("global_rules").as_array(),
                        [&](auto const& r) { return parse_rule(version, r); })
          : std::vector<rule>{};

  provider.geofencing_zones_.version_ = version;
  provider.geofencing_zones_.zones_ = std::move(zones);
  provider.geofencing_zones_.global_rules_ = std::move(global_rules);
}

}  // namespace motis::gbfs
