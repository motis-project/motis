#include <optional>
#include <string_view>
#include <vector>

#include "motis/gbfs/parser.h"

#include "cista/hash.h"

#include "utl/helpers/algorithm.h"
#include "utl/raii.h"
#include "utl/to_vec.h"

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

std::string get_localized_string(json::value const& v) {
  if (v.is_array()) {
    auto const& arr = v.as_array();
    if (!arr.empty()) {
      return static_cast<std::string>(
          arr[0].as_object().at("text").as_string());
    }
    return "";
  } else if (v.is_string()) {
    return static_cast<std::string>(v.as_string());
  } else {
    return "";
  }
}

std::string get_as_string(json::object const& obj, std::string_view const key) {
  auto const val = obj.at(key);
  if (val.is_string()) {
    return static_cast<std::string>(val.as_string());
  } else if (val.is_int64()) {
    return std::to_string(val.as_int64());
  } else if (val.is_uint64()) {
    return std::to_string(val.as_uint64());
  } else {
    return json::serialize(val);
  }
}

std::string optional_str(json::object const& obj, std::string_view key) {
  return obj.contains(key) ? get_as_string(obj, key) : "";
}

std::string optional_localized_str(json::object const& obj,
                                   std::string_view key) {
  return obj.contains(key) ? get_localized_string(obj.at(key)) : "";
}

bool get_bool(json::object const& obj,
              std::string_view const key,
              std::optional<bool> const def = std::nullopt) {
  if (!obj.contains(key) && def.has_value()) {
    return *def;
  }
  auto const val = obj.at(key);
  if (val.is_bool()) {
    return val.as_bool();
  } else if (val.is_number()) {
    return val.to_number<int>() == 1;
  } else {
    return *def;
  }
}

tg_geom* parse_multipolygon(json::object const& json) {
  utl::verify(json.at("type").as_string() == "MultiPolygon",
              "expected MultiPolygon, got {}", json.at("type").as_string());
  auto const& coordinates = json.at("coordinates").as_array();

  auto polys = std::vector<tg_poly*>{};
  UTL_FINALLY([&polys]() {
    for (auto const poly : polys) {
      tg_poly_free(poly);
    }
  })

  for (auto const& j_poly : coordinates) {
    auto rings = std::vector<tg_ring*>{};
    UTL_FINALLY([&rings]() {
      for (auto const ring : rings) {
        tg_ring_free(ring);
      }
    })
    for (auto const& j_ring : j_poly.as_array()) {
      auto points = utl::to_vec(j_ring.as_array(), [&](auto const& j_pt) {
        auto const& j_pt_arr = j_pt.as_array();
        utl::verify(j_pt_arr.size() >= 2, "invalid point in polygon ring");
        return tg_point{j_pt_arr[0].as_double(), j_pt_arr[1].as_double()};
      });
      utl::verify(points.size() > 2, "empty ring in polygon");
      // handle invalid polygons that don't have closed rings
      if (points.front().x != points.back().x ||
          points.front().y != points.back().y) {
        points.push_back(points.front());
      }
      auto ring = tg_ring_new(points.data(), static_cast<int>(points.size()));
      utl::verify(ring != nullptr, "failed to create ring");
      rings.emplace_back(ring);
    }
    utl::verify(!rings.empty(), "empty polygon in multipolygon");
    auto poly =
        tg_poly_new(rings.front(), rings.size() > 1 ? &rings[1] : nullptr,
                    static_cast<int>(rings.size() - 1));
    utl::verify(poly != nullptr, "failed to create polygon");
    polys.emplace_back(poly);
  }

  utl::verify(!polys.empty(), "empty multipolygon");
  auto const multipoly =
      tg_geom_new_multipolygon(polys.data(), static_cast<int>(polys.size()));
  utl::verify(multipoly != nullptr, "failed to create multipolygon");
  return multipoly;
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

std::optional<vehicle_type_idx_t> get_vehicle_type(
    gbfs_provider& provider,
    std::string const& vehicle_type_id,
    vehicle_start_type const start_type) {
  auto const add_vehicle_type = [&](vehicle_form_factor const ff,
                                    propulsion_type const pt,
                                    std::string const& name) {
    auto const idx = vehicle_type_idx_t{provider.vehicle_types_.size()};
    provider.vehicle_types_.emplace_back(vehicle_type{
        .id_ = vehicle_type_id,
        .idx_ = idx,
        .name_ = name,
        .form_factor_ = ff,
        .propulsion_type_ = pt,
        .return_constraint_ = provider.default_return_constraint_.value_or(
            start_type == vehicle_start_type::kStation
                ? return_constraint::kAnyStation
                : return_constraint::kFreeFloating),
        .known_return_constraint_ = false});
    provider.vehicle_types_map_[{vehicle_type_id, start_type}] = idx;
    return idx;
  };

  if (auto const it =
          provider.vehicle_types_map_.find({vehicle_type_id, start_type});
      it != end(provider.vehicle_types_map_)) {
    return it->second;
  } else if (auto const temp_it =
                 provider.temp_vehicle_types_.find(vehicle_type_id);
             temp_it != end(provider.temp_vehicle_types_)) {
    return add_vehicle_type(temp_it->second.form_factor_,
                            temp_it->second.propulsion_type_,
                            temp_it->second.name_);
  } else if (vehicle_type_id.empty()) {
    // providers that don't use vehicle types
    return add_vehicle_type(vehicle_form_factor::kBicycle,
                            propulsion_type::kHuman, "");
  }
  return {};
}

void load_system_information(gbfs_provider& provider, json::value const& root) {
  auto const& data = root.at("data").as_object();

  auto& si = provider.sys_info_;
  si.id_ = static_cast<std::string>(data.at("system_id").as_string());
  si.name_ = get_localized_string(data.at("name"));
  si.name_short_ = optional_localized_str(data, "name_short");
  si.operator_ = optional_localized_str(data, "operator");
  si.url_ = optional_str(data, "url");
  si.purchase_url_ = optional_str(data, "purchase_url");
  si.mail_ = optional_str(data, "email");
  if (data.contains("brand_assets")) {
    auto const& ba = data.at("brand_assets").as_object();
    si.color_ = optional_str(ba, "color");
  } else {
    si.color_ = "";
  }
}

void load_station_information(gbfs_provider& provider,
                              json::value const& root) {
  provider.stations_.clear();

  auto const& stations_arr = root.at("data").at("stations").as_array();
  for (auto const& s : stations_arr) {
    auto const& station_obj = s.as_object();
    auto const station_id = get_as_string(station_obj, "station_id");
    try {
      auto const name = get_localized_string(station_obj.at("name"));
      auto const lat = station_obj.at("lat").as_double();
      auto const lon = station_obj.at("lon").as_double();

      tg_geom* area = nullptr;
      if (station_obj.contains("station_area")) {
        try {
          area = parse_multipolygon(station_obj.at("station_area").as_object());
        } catch (std::exception const& ex) {
          std::cerr << "[GBFS] (" << provider.id_
                    << ") invalid station_area: " << ex.what() << "\n";
        }
      }

      provider.stations_[station_id] = station{
          .info_ = {.id_ = station_id,
                    .name_ = name,
                    .pos_ = geo::latlng{lat, lon},
                    .address_ = optional_str(station_obj, "address"),
                    .cross_street_ = optional_str(station_obj, "cross_street"),
                    .rental_uris_ = parse_rental_uris(station_obj),
                    .station_area_ =
                        std::shared_ptr<tg_geom>(area, tg_geom_deleter{})}};
    } catch (std::exception const& ex) {
      std::cerr << "[GBFS] (" << provider.id_ << ") error parsing station "
                << station_id << ": " << ex.what() << "\n";
    }
  }
}

void load_station_status(gbfs_provider& provider, json::value const& root) {
  auto const& stations_arr = root.at("data").at("stations").as_array();
  for (auto const& s : stations_arr) {
    auto const& station_obj = s.as_object();
    auto const station_id = get_as_string(station_obj, "station_id");

    auto const station_it = provider.stations_.find(station_id);
    if (station_it == end(provider.stations_)) {
      continue;
    }

    auto& station = station_it->second;
    station.status_ = station_status{
        .num_vehicles_available_ = 0U,
        .is_renting_ = get_bool(station_obj, "is_renting", true),
        .is_returning_ = get_bool(station_obj, "is_returning", true)};

    if (station_obj.contains("num_vehicles_available")) {
      // GBFS 3.x (but some 2.x feeds use this as well)
      station.status_.num_vehicles_available_ =
          station_obj.at("num_vehicles_available").to_number<unsigned>();
    } else if (station_obj.contains("num_bikes_available")) {
      // GBFS 2.x
      station.status_.num_vehicles_available_ =
          station_obj.at("num_bikes_available").to_number<unsigned>();
    }

    if (station_obj.contains("vehicle_types_available")) {
      auto const& vta = station_obj.at("vehicle_types_available").as_array();
      auto unrestricted_available = 0U;
      auto any_station_available = 0U;
      auto roundtrip_available = 0U;
      for (auto const& vt : vta) {
        auto const vehicle_type_id =
            static_cast<std::string>(vt.at("vehicle_type_id").as_string());
        auto const count = vt.at("count").to_number<unsigned>();
        if (auto const vt_idx = get_vehicle_type(provider, vehicle_type_id,
                                                 vehicle_start_type::kStation);
            vt_idx) {
          station.status_.vehicle_types_available_[*vt_idx] = count;
          switch (provider.vehicle_types_[*vt_idx].return_constraint_) {
            case return_constraint::kFreeFloating:
              unrestricted_available += count;
              break;
            case return_constraint::kAnyStation:
              any_station_available += count;
              break;
            case return_constraint::kRoundtripStation:
              roundtrip_available += count;
              break;
          }
        }
      }
      station.status_.num_vehicles_available_ =
          unrestricted_available + any_station_available + roundtrip_available;
    } else {
      if (auto const vt_idx =
              get_vehicle_type(provider, "", vehicle_start_type::kStation);
          vt_idx) {
        station.status_.vehicle_types_available_[*vt_idx] =
            station.status_.num_vehicles_available_;
      }
    }

    if (station_obj.contains("vehicle_docks_available")) {
      for (auto const& vt :
           station_obj.at("vehicle_docks_available").as_array()) {
        auto& vto = vt.as_object();
        if (vto.contains("vehicle_type_ids") && vto.contains("count")) {
          for (auto const& vti : vto.at("vehicle_type_ids").as_array()) {
            auto const vehicle_type_id =
                static_cast<std::string>(vti.as_string());
            if (auto const vt_idx = get_vehicle_type(
                    provider, vehicle_type_id, vehicle_start_type::kStation);
                vt_idx) {
              station.status_.vehicle_docks_available_[*vt_idx] =
                  vto.at("count").to_number<unsigned>();
            }
          }
        }
      }
    }
  }
}

vehicle_form_factor parse_form_factor(std::string_view const s) {
  switch (cista::hash(s)) {
    case cista::hash("bicycle"):
    case cista::hash("bike"):  // non-standard
      return vehicle_form_factor::kBicycle;
    case cista::hash("cargo_bicycle"):
      return vehicle_form_factor::kCargoBicycle;
    case cista::hash("car"): return vehicle_form_factor::kCar;
    case cista::hash("moped"): return vehicle_form_factor::kMoped;
    case cista::hash("scooter"):  // < 3.0
    case cista::hash("scooter_standing"):
      return vehicle_form_factor::kScooterStanding;
    case cista::hash("scooter_seated"):
      return vehicle_form_factor::kScooterSeated;
    case cista::hash("other"):
    default: return vehicle_form_factor::kOther;
  }
}

propulsion_type parse_propulsion_type(std::string_view const s) {
  switch (cista::hash(s)) {
    case cista::hash("human"): return propulsion_type::kHuman;
    case cista::hash("electric_assist"):
      return propulsion_type::kElectricAssist;
    case cista::hash("electric"): return propulsion_type::kElectric;
    case cista::hash("combustion"): return propulsion_type::kCombustion;
    case cista::hash("combustion_diesel"):
      return propulsion_type::kCombustionDiesel;
    case cista::hash("hybrid"): return propulsion_type::kHybrid;
    case cista::hash("plug_in_hybrid"): return propulsion_type::kPlugInHybrid;
    case cista::hash("hydrogen_fuel_cell"):
      return propulsion_type::kHydrogenFuelCell;
    default: return propulsion_type::kHuman;
  }
}

std::optional<return_constraint> parse_return_constraint(
    std::string_view const s) {
  switch (cista::hash(s)) {
    case cista::hash("any_station"): return return_constraint::kAnyStation;
    case cista::hash("roundtrip_station"):
      return return_constraint::kRoundtripStation;
    case cista::hash("free_floating"):
    case cista::hash("hybrid"): return return_constraint::kFreeFloating;
    default: return {};
  }
}

std::optional<return_constraint> parse_return_constraint(
    json::object const& vt) {
  if (vt.contains("return_constraint")) {
    return parse_return_constraint(vt.at("return_constraint").as_string());
  }
  return {};
}

void load_vehicle_types(gbfs_provider& provider, json::value const& root) {
  provider.vehicle_types_.clear();
  provider.vehicle_types_map_.clear();
  provider.temp_vehicle_types_.clear();
  for (auto const& v : root.at("data").at("vehicle_types").as_array()) {
    auto const id =
        static_cast<std::string>(v.at("vehicle_type_id").as_string());
    auto const name = optional_localized_str(v.as_object(), "name");
    auto const rc = parse_return_constraint(v.as_object());
    auto const form_factor =
        parse_form_factor(optional_str(v.as_object(), "form_factor"));
    auto const propulsion_type =
        parse_propulsion_type(optional_str(v.as_object(), "propulsion_type"));
    if (rc) {
      auto const idx = vehicle_type_idx_t{provider.vehicle_types_.size()};
      provider.vehicle_types_.emplace_back(
          vehicle_type{.id_ = id,
                       .idx_ = idx,
                       .name_ = name,
                       .form_factor_ = form_factor,
                       .propulsion_type_ = propulsion_type,
                       .return_constraint_ = *rc,
                       .known_return_constraint_ = true});
      provider.vehicle_types_map_[{id, vehicle_start_type::kStation}] = idx;
      provider.vehicle_types_map_[{id, vehicle_start_type::kFreeFloating}] =
          idx;
    } else {
      provider.temp_vehicle_types_[id] = temp_vehicle_type{
          .id_ = id,
          .name_ = name,
          .form_factor_ = form_factor,
          .propulsion_type_ = propulsion_type,
      };
    }
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

    auto pos = geo::latlng{};
    if (vehicle_obj.contains("lat") && vehicle_obj.contains("lon")) {
      auto const lat = vehicle_obj.at("lat");
      auto const lon = vehicle_obj.at("lon");
      if (!lat.is_double() || !lon.is_double()) {
        continue;
      }
      pos = geo::latlng{vehicle_obj.at("lat").as_double(),
                        vehicle_obj.at("lon").as_double()};
    } else if (vehicle_obj.contains("station_id")) {
      auto const station_id = get_as_string(vehicle_obj, "station_id");
      if (auto const it = provider.stations_.find(station_id);
          it != end(provider.stations_)) {
        pos = it->second.info_.pos_;
      } else {
        continue;
      }
    } else {
      continue;
    }

    auto const id = get_as_string(
        vehicle_obj, version == gbfs_version::k3 ? "vehicle_id" : "bike_id");

    auto const type_id = optional_str(vehicle_obj, "vehicle_type_id");
    auto type_idx = vehicle_type_idx_t::invalid();

    if (auto const vt_idx = get_vehicle_type(provider, type_id,
                                             vehicle_start_type::kFreeFloating);
        vt_idx) {
      type_idx = *vt_idx;
    }

    provider.vehicle_status_.emplace_back(vehicle_status{
        .id_ = id,
        .pos_ = pos,
        .is_reserved_ = get_bool(vehicle_obj, "is_reserved", false),
        .is_disabled_ = get_bool(vehicle_obj, "is_disabled", false),
        .vehicle_type_idx_ = type_idx,
        .station_id_ = optional_str(vehicle_obj, "station_id"),
        .home_station_id_ = optional_str(vehicle_obj, "home_station_id"),
        .rental_uris_ = parse_rental_uris(vehicle_obj)});
  }

  utl::sort(provider.vehicle_status_);
}

rule parse_rule(gbfs_provider& provider,
                gbfs_version const version,
                json::value const& r) {
  auto const vti_key =
      version == gbfs_version::k2 ? "vehicle_type_id" : "vehicle_type_ids";
  auto const& rule_obj = r.as_object();

  auto vehicle_type_idxs = std::vector<vehicle_type_idx_t>{};
  if (rule_obj.contains(vti_key)) {
    for (auto const& vt : rule_obj.at(vti_key).as_array()) {
      auto const vt_id = static_cast<std::string>(vt.as_string());
      if (auto const it = provider.vehicle_types_map_.find(
              {vt_id, vehicle_start_type::kStation});
          it != end(provider.vehicle_types_map_)) {
        vehicle_type_idxs.emplace_back(it->second);
      }
      if (auto const it = provider.vehicle_types_map_.find(
              {vt_id, vehicle_start_type::kFreeFloating});
          it != end(provider.vehicle_types_map_)) {
        vehicle_type_idxs.emplace_back(it->second);
      }
    }
  }

  return rule{
      .vehicle_type_idxs_ = std::move(vehicle_type_idxs),
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

  auto zones = std::vector<zone>{};
  auto const zones_arr = zones_obj.at("features").as_array();
  zones.reserve(zones_arr.size());
  for (auto const& z : zones_arr) {
    try {
      auto const& props = z.at("properties").as_object();
      if (!props.contains("rules") || !props.at("rules").is_array()) {
        continue;
      }
      auto rules = utl::to_vec(
          props.at("rules").as_array(),
          [&](auto const& r) { return parse_rule(provider, version, r); });

      auto* geom = parse_multipolygon(z.at("geometry").as_object());

      auto name = optional_localized_str(props, "name");

      zones.emplace_back(geom, std::move(rules), std::move(name));
    } catch (std::exception const& ex) {
      std::cerr << "[GBFS] (" << provider.id_
                << ") invalid geofencing zone: " << ex.what() << "\n";
    }
  }

  //  required in 3.0, but some feeds don't have it
  auto global_rules =
      root.at("data").as_object().contains("global_rules") &&
              root.at("data").at("global_rules").is_array()
          ? utl::to_vec(
                root.at("data").at("global_rules").as_array(),
                [&](auto const& r) { return parse_rule(provider, version, r); })
          : std::vector<rule>{};

  provider.geofencing_zones_.version_ = version;
  provider.geofencing_zones_.zones_ = std::move(zones);
  provider.geofencing_zones_.global_rules_ = std::move(global_rules);
}

}  // namespace motis::gbfs
