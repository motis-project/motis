#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

#include "date/date.h"

#include "motis/gbfs/parser.h"

#include "cista/hash.h"

#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"
#include "utl/raii.h"
#include "utl/to_vec.h"

namespace json = boost::json;

namespace motis::gbfs {

std::optional<std::string> as_string(json::value const& v) {
  if (v.is_string()) {
    return static_cast<std::string>(v.as_string());
  } else if (v.is_int64()) {
    return std::to_string(v.as_int64());
  } else if (v.is_uint64()) {
    return std::to_string(v.as_uint64());
  } else if (v.is_double()) {
    return std::to_string(v.as_double());
  }
  return std::nullopt;
}

std::optional<double> as_double(json::value const& v) {
  if (v.is_double()) {
    return v.as_double();
  } else if (v.is_int64()) {
    return static_cast<double>(v.as_int64());
  } else if (v.is_uint64()) {
    return static_cast<double>(v.as_uint64());
  } else if (v.is_string()) {
    auto const s = std::string{v.as_string()};
    auto c = utl::cstr{s.c_str(), s.size()};
    auto d = double{};
    if (utl::parse_fp(c, d) && c.len == 0) {
      return d;
    }
  }
  return std::nullopt;
}

std::optional<unsigned> as_count(json::value const& v) {
  if (auto const d = as_double(v); d) {
    return static_cast<unsigned>(std::max(0.0, *d));
  }
  return std::nullopt;
}

std::optional<std::chrono::system_clock::time_point> parse_timestamp(
    json::value const& val) {
  if (val.is_int64()) {
    return std::chrono::system_clock::time_point{
        std::chrono::seconds{val.to_number<std::int64_t>()}};
  }
  if (val.is_uint64()) {
    return std::chrono::system_clock::time_point{std::chrono::seconds{
        static_cast<std::int64_t>(val.to_number<std::uint64_t>())}};
  }
  if (val.is_string()) {
    auto const s = std::string{val.as_string()};
    auto* end = static_cast<char*>(nullptr);
    auto const ts = std::strtoll(s.c_str(), &end, 10);
    if (end != s.c_str() && *end == '\0') {
      return std::chrono::system_clock::time_point{std::chrono::seconds{ts}};
    }

    auto tp = std::chrono::system_clock::time_point{};
    auto iss = std::istringstream{s};
    if (iss >> date::parse("%FT%T%Ez", tp); !iss.fail()) {
      return tp;
    }
    iss.clear();
    iss.str(s);
    if (iss >> date::parse("%FT%T%z", tp); !iss.fail()) {
      return tp;
    }
    iss.clear();
    iss.str(s);
    if (iss >> date::parse("%FT%T", tp); !iss.fail()) {
      return tp;
    }
  }
  return std::nullopt;
}

std::string optional_str(json::object const& obj, std::string_view key) {
  auto const it = obj.find(key);
  if (it == obj.end()) {
    return "";
  }
  return as_string(it->value()).value_or("");
}

std::string get_localized_string(json::value const& v) {
  if (v.is_array()) {
    for (auto const& entry : v.as_array()) {
      if (!entry.is_object()) {
        continue;
      }
      auto const text = optional_str(entry.as_object(), "text");
      if (!text.empty()) {
        return text;
      }
    }
    return "";
  } else if (v.is_string()) {
    return static_cast<std::string>(v.as_string());
  } else {
    return "";
  }
}

std::string optional_localized_str(json::object const& obj,
                                   std::string_view key) {
  return obj.contains(key) ? get_localized_string(obj.at(key)) : "";
}

bool get_bool(json::object const& obj,
              std::string_view const key,
              bool const def) {
  auto const it = obj.find(key);
  if (it == obj.end()) {
    return def;
  }
  auto const& val = it->value();
  if (val.is_bool()) {
    return val.as_bool();
  } else if (val.is_number()) {
    return val.to_number<int>() == 1;
  } else if (val.is_string()) {
    auto const s = std::string_view{val.as_string()};
    if (s == "true" || s == "1" || s == "yes") {
      return true;
    }
    if (s == "false" || s == "0" || s == "no") {
      return false;
    }
    return def;
  } else {
    return def;
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
        auto const x = as_double(j_pt_arr[0]);
        auto const y = as_double(j_pt_arr[1]);
        utl::verify(x.has_value() && y.has_value(),
                    "invalid point in polygon ring");
        return tg_point{*x, *y};
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

  auto const* feeds = static_cast<json::array const*>(nullptr);
  if (auto const it = data.find("feeds");
      it != data.end() && it->value().is_array()) {
    feeds = &it->value().as_array();
  } else {
    for (auto const& [_, lang] : data) {
      if (!lang.is_object()) {
        continue;
      }
      auto const& lang_obj = lang.as_object();
      if (auto const feeds_it = lang_obj.find("feeds");
          feeds_it != lang_obj.end() && feeds_it->value().is_array()) {
        feeds = &feeds_it->value().as_array();
        break;
      }
    }
  }

  if (feeds == nullptr) {
    return urls;
  }

  for (auto const& feed : *feeds) {
    if (!feed.is_object()) {
      continue;
    }
    auto const& feed_obj = feed.as_object();
    auto const name = optional_str(feed_obj, "name");
    auto const url = optional_str(feed_obj, "url");
    if (name.empty() || url.empty()) {
      continue;
    }
    urls[name] = url;
  }
  return urls;
}

rental_uris parse_rental_uris(json::object const& parent) {
  auto uris = rental_uris{};

  if (auto const it = parent.find("rental_uris");
      it != parent.end() && it->value().is_object()) {
    auto const& o = it->value().as_object();
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
  si.id_ = optional_str(data, "system_id");
  if (si.id_.empty()) {
    si.id_ = provider.id_;
  }
  si.name_ = optional_localized_str(data, "name");
  if (si.name_.empty()) {
    si.name_ = provider.id_;
  }
  si.name_short_ = optional_localized_str(data, "name_short");
  si.operator_ = optional_localized_str(data, "operator");
  si.url_ = optional_str(data, "url");
  si.purchase_url_ = optional_str(data, "purchase_url");
  si.mail_ = optional_str(data, "email");
  if (auto const it = data.find("brand_assets");
      it != data.end() && it->value().is_object()) {
    auto const& ba = it->value().as_object();
    si.color_ = optional_str(ba, "color");
  } else {
    si.color_ = "";
  }
}

void load_station_information(gbfs_provider& provider,
                              json::value const& root) {
  auto const& stations_arr = root.at("data").at("stations").as_array();

  provider.stations_.clear();

  for (auto const& s : stations_arr) {
    if (!s.is_object()) {
      ++provider.skipped_station_infos_;
      continue;
    }
    auto const& station_obj = s.as_object();
    auto const station_id = optional_str(station_obj, "station_id");
    auto const lat = station_obj.contains("lat")
                         ? as_double(station_obj.at("lat"))
                         : std::nullopt;
    auto const lon = station_obj.contains("lon")
                         ? as_double(station_obj.at("lon"))
                         : std::nullopt;
    if (station_id.empty() || !lat || !lon) {
      ++provider.skipped_station_infos_;
      continue;
    }

    auto const name = optional_localized_str(station_obj, "name");

    tg_geom* area = nullptr;
    if (auto const it = station_obj.find("station_area");
        it != station_obj.end() && it->value().is_object()) {
      try {
        area = parse_multipolygon(it->value().as_object());
      } catch (std::exception const& ex) {
        std::cerr << "[GBFS] (" << provider.id_
                  << ") invalid station_area: " << ex.what() << "\n";
      }
    }

    provider.stations_[station_id] = station{
        .info_ = {.id_ = station_id,
                  .name_ = name,
                  .pos_ = geo::latlng{*lat, *lon},
                  .address_ = optional_str(station_obj, "address"),
                  .cross_street_ = optional_str(station_obj, "cross_street"),
                  .rental_uris_ = parse_rental_uris(station_obj),
                  .station_area_ =
                      std::shared_ptr<tg_geom>(area, tg_geom_deleter{})}};
  }
}

void load_station_status(gbfs_provider& provider, json::value const& root) {
  auto const& stations_arr = root.at("data").at("stations").as_array();

  for (auto const& s : stations_arr) {
    if (!s.is_object()) {
      ++provider.skipped_station_status_;
      continue;
    }
    auto const& station_obj = s.as_object();
    auto const station_id = optional_str(station_obj, "station_id");

    auto const station_it = provider.stations_.find(station_id);
    if (station_it == end(provider.stations_)) {
      ++provider.skipped_station_status_;
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
          as_count(station_obj.at("num_vehicles_available")).value_or(0U);
    } else if (station_obj.contains("num_bikes_available")) {
      // GBFS 2.x
      station.status_.num_vehicles_available_ =
          as_count(station_obj.at("num_bikes_available")).value_or(0U);
    }

    if (auto const it = station_obj.find("vehicle_types_available");
        it != station_obj.end() && it->value().is_array()) {
      auto const& vta = it->value().as_array();
      auto unrestricted_available = 0U;
      auto any_station_available = 0U;
      auto roundtrip_available = 0U;
      auto has_typed_availability = false;
      for (auto const& vt : vta) {
        if (!vt.is_object()) {
          ++provider.skipped_station_status_;
          continue;
        }
        auto const& vt_obj = vt.as_object();
        auto const vehicle_type_id = optional_str(vt_obj, "vehicle_type_id");
        if (vehicle_type_id.empty()) {
          ++provider.skipped_station_status_;
          continue;
        }
        auto const count = vt_obj.contains("count")
                               ? as_count(vt_obj.at("count")).value_or(0U)
                               : 0U;
        if (auto const vt_idx = get_vehicle_type(provider, vehicle_type_id,
                                                 vehicle_start_type::kStation);
            vt_idx) {
          has_typed_availability = true;
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
        } else {
          ++provider.skipped_station_status_;
        }
      }
      if (has_typed_availability) {
        station.status_.num_vehicles_available_ = unrestricted_available +
                                                  any_station_available +
                                                  roundtrip_available;
      }
    } else {
      if (auto const vt_idx =
              get_vehicle_type(provider, "", vehicle_start_type::kStation);
          vt_idx) {
        station.status_.vehicle_types_available_[*vt_idx] =
            station.status_.num_vehicles_available_;
      }
    }

    if (auto const it = station_obj.find("vehicle_docks_available");
        it != station_obj.end() && it->value().is_array()) {
      for (auto const& vt : it->value().as_array()) {
        if (!vt.is_object()) {
          ++provider.skipped_station_status_;
          continue;
        }
        auto const& vto = vt.as_object();
        if (!vto.contains("vehicle_type_ids") || !vto.contains("count") ||
            !vto.at("vehicle_type_ids").is_array()) {
          ++provider.skipped_station_status_;
          continue;
        }
        auto const count = as_count(vto.at("count")).value_or(0U);
        for (auto const& vti : vto.at("vehicle_type_ids").as_array()) {
          auto const vehicle_type_id = as_string(vti).value_or("");
          if (vehicle_type_id.empty()) {
            ++provider.skipped_station_status_;
            continue;
          }
          if (auto const vt_idx = get_vehicle_type(
                  provider, vehicle_type_id, vehicle_start_type::kStation);
              vt_idx) {
            station.status_.vehicle_docks_available_[*vt_idx] = count;
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
    return parse_return_constraint(optional_str(vt, "return_constraint"));
  }
  return {};
}

void load_vehicle_types(gbfs_provider& provider, json::value const& root) {
  auto const& vehicle_types = root.at("data").at("vehicle_types").as_array();

  provider.vehicle_types_.clear();
  provider.vehicle_types_map_.clear();
  provider.temp_vehicle_types_.clear();

  for (auto const& v : vehicle_types) {
    if (!v.is_object()) {
      ++provider.skipped_vehicle_types_;
      continue;
    }
    auto const& vt_obj = v.as_object();
    auto const id = optional_str(vt_obj, "vehicle_type_id");
    if (id.empty()) {
      ++provider.skipped_vehicle_types_;
      continue;
    }
    auto const name = optional_localized_str(vt_obj, "name");
    auto const rc = parse_return_constraint(vt_obj);
    auto const form_factor =
        parse_form_factor(optional_str(vt_obj, "form_factor"));
    auto const propulsion_type =
        parse_propulsion_type(optional_str(vt_obj, "propulsion_type"));
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
  auto const& data = root.at("data").as_object();
  auto const get_array = [&](std::string_view const key) {
    auto const it = data.find(key);
    return it != data.end() && it->value().is_array() ? &it->value().as_array()
                                                      : nullptr;
  };
  auto const* vehicles_arr = get_array("vehicles");
  vehicles_arr = vehicles_arr != nullptr ? vehicles_arr : get_array("bikes");
  utl::verify(vehicles_arr != nullptr, "missing vehicles/bikes array");

  provider.vehicle_status_.clear();

  for (auto const& v : *vehicles_arr) {
    if (!v.is_object()) {
      ++provider.skipped_vehicle_status_;
      continue;
    }
    auto const& vehicle_obj = v.as_object();

    auto pos = geo::latlng{};
    if (vehicle_obj.contains("lat") && vehicle_obj.contains("lon")) {
      auto const lat = as_double(vehicle_obj.at("lat"));
      auto const lon = as_double(vehicle_obj.at("lon"));
      if (!lat || !lon) {
        ++provider.skipped_vehicle_status_;
        continue;
      }
      pos = geo::latlng{*lat, *lon};
    } else if (vehicle_obj.contains("station_id")) {
      auto const station_id = optional_str(vehicle_obj, "station_id");
      if (auto const it = provider.stations_.find(station_id);
          it != end(provider.stations_)) {
        pos = it->second.info_.pos_;
      } else {
        ++provider.skipped_vehicle_status_;
        continue;
      }
    } else {
      ++provider.skipped_vehicle_status_;
      continue;
    }

    auto id = optional_str(vehicle_obj, "vehicle_id");
    if (id.empty()) {
      id = optional_str(vehicle_obj, "bike_id");
    }
    if (id.empty()) {
      ++provider.skipped_vehicle_status_;
      continue;
    }

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

rule parse_rule(gbfs_provider& provider, json::value const& r) {
  auto const& rule_obj = r.as_object();

  auto vehicle_type_idxs = std::vector<vehicle_type_idx_t>{};
  auto const parse_vehicle_type_ids = [](json::value const& val) {
    auto vehicle_type_ids = std::vector<std::string>{};
    if (val.is_array()) {
      for (auto const& vt : val.as_array()) {
        if (auto const vt_id = as_string(vt); vt_id) {
          vehicle_type_ids.emplace_back(*vt_id);
        }
      }
    } else if (auto const vt_id = as_string(val); vt_id) {
      vehicle_type_ids.emplace_back(*vt_id);
    }
    return vehicle_type_ids;
  };

  auto vehicle_type_ids = std::vector<std::string>{};
  if (auto const it = rule_obj.find("vehicle_type_ids"); it != rule_obj.end()) {
    vehicle_type_ids = parse_vehicle_type_ids(it->value());
  }
  if (vehicle_type_ids.empty()) {
    if (auto const it = rule_obj.find("vehicle_type_id");
        it != rule_obj.end()) {
      vehicle_type_ids = parse_vehicle_type_ids(it->value());
    }
  }

  for (auto const& vt_id : vehicle_type_ids) {
    if (auto const station_vt_it = provider.vehicle_types_map_.find(
            {vt_id, vehicle_start_type::kStation});
        station_vt_it != end(provider.vehicle_types_map_)) {
      vehicle_type_idxs.emplace_back(station_vt_it->second);
    }
    if (auto const free_floating_vt_it = provider.vehicle_types_map_.find(
            {vt_id, vehicle_start_type::kFreeFloating});
        free_floating_vt_it != end(provider.vehicle_types_map_)) {
      vehicle_type_idxs.emplace_back(free_floating_vt_it->second);
    }
  }
  if (!vehicle_type_ids.empty() && vehicle_type_idxs.empty()) {
    vehicle_type_idxs.emplace_back(vehicle_type_idx_t::invalid());
  }

  auto const ride_allowed = get_bool(rule_obj, "ride_allowed", true);
  return rule{
      .vehicle_type_idxs_ = std::move(vehicle_type_idxs),
      .ride_start_allowed_ =
          get_bool(rule_obj, "ride_start_allowed", ride_allowed),
      .ride_end_allowed_ = get_bool(rule_obj, "ride_end_allowed", ride_allowed),
      .ride_through_allowed_ =
          get_bool(rule_obj, "ride_through_allowed", ride_allowed),
      .station_parking_ =
          rule_obj.contains("station_parking")
              ? std::optional{get_bool(rule_obj, "station_parking", false)}
              : std::nullopt};
}

void load_geofencing_zones(gbfs_provider& provider, json::value const& root) {
  auto const& data = root.at("data").as_object();
  auto const it = data.find("geofencing_zones");
  utl::verify(it != data.end() && it->value().is_object(),
              "missing geofencing_zones object");
  auto const& zones_obj = it->value().as_object();
  utl::verify(optional_str(zones_obj, "type") == "FeatureCollection",
              "geofencing_zones is not a FeatureCollection");

  auto zones = std::vector<zone>{};
  auto const features_it = zones_obj.find("features");
  utl::verify(features_it != zones_obj.end() && features_it->value().is_array(),
              "missing geofencing_zones features array");
  auto const zones_arr = features_it->value().as_array();
  zones.reserve(zones_arr.size());
  for (auto const& z : zones_arr) {
    try {
      if (!z.is_object()) {
        ++provider.skipped_geofencing_zones_;
        continue;
      }
      auto const& props = z.at("properties").as_object();
      if (!props.contains("rules") || !props.at("rules").is_array()) {
        ++provider.skipped_geofencing_zones_;
        continue;
      }
      auto rules = std::vector<rule>{};
      for (auto const& r : props.at("rules").as_array()) {
        try {
          if (r.is_object()) {
            rules.emplace_back(parse_rule(provider, r));
          } else {
            ++provider.skipped_geofencing_rules_;
          }
        } catch (std::exception const& ex) {
          ++provider.skipped_geofencing_rules_;
          std::cerr << "[GBFS] (" << provider.id_
                    << ") invalid geofencing rule: " << ex.what() << "\n";
        }
      }
      if (rules.empty()) {
        ++provider.skipped_geofencing_zones_;
        continue;
      }

      auto* geom = parse_multipolygon(z.at("geometry").as_object());

      auto name = optional_localized_str(props, "name");

      zones.emplace_back(geom, std::move(rules), std::move(name));
    } catch (std::exception const& ex) {
      ++provider.skipped_geofencing_zones_;
      std::cerr << "[GBFS] (" << provider.id_
                << ") invalid geofencing zone: " << ex.what() << "\n";
    }
  }

  //  required in 3.0, but some feeds don't have it
  auto global_rules = std::vector<rule>{};
  if (auto const global_rules_it = data.find("global_rules");
      global_rules_it != data.end() && global_rules_it->value().is_array()) {
    for (auto const& r : global_rules_it->value().as_array()) {
      try {
        if (r.is_object()) {
          global_rules.emplace_back(parse_rule(provider, r));
        } else {
          ++provider.skipped_geofencing_rules_;
        }
      } catch (std::exception const& ex) {
        ++provider.skipped_geofencing_rules_;
        std::cerr << "[GBFS] (" << provider.id_
                  << ") invalid global geofencing rule: " << ex.what() << "\n";
      }
    }
  }

  provider.geofencing_zones_.zones_ = std::move(zones);
  provider.geofencing_zones_.global_rules_ = std::move(global_rules);
}

}  // namespace motis::gbfs
