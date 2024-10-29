#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "tg.h"

#include "cista/strong.h"

#include "geo/latlng.h"

#include "osr/routing/additional_edge.h"
#include "osr/types.h"

#include "motis/point_rtree.h"
#include "motis/types.h"

namespace motis::gbfs {

enum class gbfs_version : std::uint8_t {
  k1 = 0,
  k2 = 1,
  k3 = 2,
};

enum class vehicle_form_factor : std::uint8_t {
  kBicycle = 0,
  kCargoBicycle = 1,
  kCar = 2,
  kMoped = 3,
  kScooterStanding = 4,
  kScooterSeated = 5,
  kOther = 6
};

enum class return_constraint : std::uint8_t {
  kNone = 0,  // includes free_floating + hybrid
  kAnyStation = 1,
  kRoundtripStation = 2
};

struct vehicle_type {
  std::string id_;
  vehicle_form_factor form_factor_{};
  return_constraint return_constraint_{};
};

struct system_information {
  std::string id_;
  std::string name_;
  std::string name_short_;
  std::string operator_;
  std::string url_;
  std::string purchase_url_;
  std::string mail_;
};

struct rental_uris {
  // all fields are optional
  std::string android_;
  std::string ios_;
  std::string web_;
};

struct tg_geom_deleter {
  void operator()(tg_geom* ptr) const {
    if (ptr != nullptr) {
      tg_geom_free(ptr);
    }
  }
};

struct station_information {
  std::string id_;
  std::string name_;
  geo::latlng pos_{};
  // optional:
  std::string address_{};
  std::string cross_street_{};
  rental_uris rental_uris_{};

  std::unique_ptr<tg_geom, tg_geom_deleter> station_area_{};
};

struct station_status {
  unsigned num_vehicles_available_{};
  hash_map<std::string, unsigned> vehicle_types_available_{};
  bool is_renting_{true};
  bool is_returning_{true};
};

struct station {
  station_information info_{};
  station_status status_{};
};

struct vehicle_status {
  std::string id_;
  geo::latlng pos_;
  bool is_reserved_{};
  bool is_disabled_{};
  std::string vehicle_type_id_;
  std::string station_id_;
  std::string home_station_id_;
  rental_uris rental_uris_{};
};

struct rule {
  std::vector<std::string> vehicle_type_ids_{};
  bool ride_start_allowed_{};
  bool ride_end_allowed_{};
  bool ride_through_allowed_{};
  std::optional<bool> station_parking_{};
};

struct geofencing_restrictions {
  bool ride_start_allowed_{true};
  bool ride_end_allowed_{true};
  bool ride_through_allowed_{true};
  std::optional<bool> station_parking_{};
};

struct zone {
  zone() = default;
  zone(tg_geom* geom, std::vector<rule>&& rules, std::string&& name)
      : geom_{geom},
        rules_{std::move(rules)},
        clockwise_{geom_ && tg_geom_num_polys(geom_.get()) > 0
                       ? tg_poly_clockwise(tg_geom_poly_at(geom_.get(), 0))
                       : true},
        name_{std::move(name)} {}

  std::unique_ptr<tg_geom, tg_geom_deleter> geom_;
  std::vector<rule> rules_;
  bool clockwise_{true};
  std::string name_;
};

struct geofencing_zones {
  gbfs_version version_{};
  std::vector<zone> zones_;
  std::vector<rule> global_rules_;

  void clear();
  geofencing_restrictions get_restrictions(
      geo::latlng const&, geofencing_restrictions const& default_restrictions);
};

struct additional_node {
  struct station {
    std::string id_;
  };

  struct vehicle {
    std::size_t idx_{};
  };

  std::variant<station, vehicle> data_;
};

struct gbfs_provider {
  std::string id_;  // from config
  gbfs_provider_idx_t idx_{};

  system_information sys_info_{};
  hash_map<std::string, station> stations_{};
  hash_map<std::string, vehicle_type> vehicle_types_{};
  std::vector<vehicle_status> vehicle_status_;
  geofencing_zones geofencing_zones_{};
  geofencing_restrictions default_restrictions_{};

  std::vector<additional_node> additional_nodes_;
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>
      additional_edges_;

  osr::bitvec<osr::node_idx_t> start_allowed_;
  osr::bitvec<osr::node_idx_t> end_allowed_;
  osr::bitvec<osr::node_idx_t> through_allowed_;

  bool has_vehicles_to_rent_{};
};

struct gbfs_data {
  vector_map<gbfs_provider_idx_t, gbfs_provider> providers_{};
  hash_map<std::string, gbfs_provider_idx_t> provider_by_id_{};
  point_rtree<gbfs_provider_idx_t> provider_rtree_{};
};

}  // namespace motis::gbfs
