#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tg.h"

#include "cista/hash.h"
#include "cista/strong.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "osr/routing/additional_edge.h"
#include "osr/routing/sharing_data.h"
#include "osr/types.h"

#include "motis/box_rtree.h"
#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

#include "motis/gbfs/lru_cache.h"

namespace motis::gbfs {

enum class gbfs_version : std::uint8_t {
  k1 = 0,
  k2 = 1,
  k3 = 2,
};

using vehicle_type_idx_t =
    cista::strong<std::uint16_t, struct vehicle_type_idx_>;

enum class vehicle_form_factor : std::uint8_t {
  kBicycle = 0,
  kCargoBicycle = 1,
  kCar = 2,
  kMoped = 3,
  kScooterStanding = 4,
  kScooterSeated = 5,
  kOther = 6
};

enum class propulsion_type : std::uint8_t {
  kHuman = 0,
  kElectricAssist = 1,
  kElectric = 2,
  kCombustion = 3,
  kCombustionDiesel = 4,
  kHybrid = 5,
  kPlugInHybrid = 6,
  kHydrogenFuelCell = 7
};

enum class return_constraint : std::uint8_t {
  kFreeFloating = 0,  // includes hybrid
  kAnyStation = 1,
  kRoundtripStation = 2
};

struct vehicle_type {
  std::string id_{};
  vehicle_type_idx_t idx_{vehicle_type_idx_t::invalid()};
  std::string name_{};
  vehicle_form_factor form_factor_{};
  propulsion_type propulsion_type_{};
  return_constraint return_constraint_{};
  bool known_return_constraint_{};  // true if taken from feed, false if guessed
};

struct temp_vehicle_type {
  std::string id_;
  std::string name_;
  vehicle_form_factor form_factor_{};
  propulsion_type propulsion_type_{};
};

enum class vehicle_start_type : std::uint8_t {
  kStation = 0,
  kFreeFloating = 1
};

struct system_information {
  std::string id_;
  std::string name_;
  std::string name_short_;
  std::string operator_;
  std::string url_;
  std::string purchase_url_;
  std::string mail_;
  std::string color_;
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

  std::shared_ptr<tg_geom> station_area_{};

  geo::box bounding_box() const {
    if (station_area_) {
      auto const rect = tg_geom_rect(station_area_.get());
      return geo::box{geo::latlng{rect.min.y, rect.min.x},
                      geo::latlng{rect.max.y, rect.max.x}};
    } else {
      return geo::box{pos_, pos_};
    }
  }
};

struct station_status {
  unsigned num_vehicles_available_{};
  hash_map<vehicle_type_idx_t, unsigned> vehicle_types_available_{};
  hash_map<vehicle_type_idx_t, unsigned> vehicle_docks_available_{};
  bool is_renting_{true};
  bool is_returning_{true};
};

struct station {
  station_information info_{};
  station_status status_{};
};

struct vehicle_status {
  bool operator==(vehicle_status const& o) const { return id_ == o.id_; }
  auto operator<=>(vehicle_status const& o) const { return id_ <=> o.id_; }

  std::string id_;
  geo::latlng pos_;
  bool is_reserved_{};
  bool is_disabled_{};
  vehicle_type_idx_t vehicle_type_idx_;
  std::string station_id_;
  std::string home_station_id_;
  rental_uris rental_uris_{};
};

struct rule {
  std::vector<vehicle_type_idx_t> vehicle_type_idxs_{};
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
      : geom_{geom, tg_geom_deleter{}},
        rules_{std::move(rules)},
        clockwise_{geom_ && tg_geom_num_polys(geom_.get()) > 0
                       ? tg_poly_clockwise(tg_geom_poly_at(geom_.get(), 0))
                       : true},
        name_{std::move(name)} {}

  geo::box bounding_box() const {
    auto const rect = tg_geom_rect(geom_.get());
    return geo::box{geo::latlng{rect.min.y, rect.min.x},
                    geo::latlng{rect.max.y, rect.max.x}};
  }

  std::shared_ptr<tg_geom> geom_;
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
      geo::latlng const& pos,
      vehicle_type_idx_t,
      geofencing_restrictions const& default_restrictions) const;
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

struct file_info {
  bool has_value() const { return expiry_.has_value(); }

  bool needs_update(std::chrono::system_clock::time_point const now) const {
    return !expiry_.has_value() || *expiry_ < now;
  }

  std::optional<std::chrono::system_clock::time_point> expiry_{};
  cista::hash_t hash_{};
};

struct provider_file_infos {
  bool needs_update() const {
    auto const now = std::chrono::system_clock::now();
    return urls_fi_.needs_update(now) ||
           system_information_fi_.needs_update(now) ||
           vehicle_types_fi_.needs_update(now) ||
           station_information_fi_.needs_update(now) ||
           station_status_fi_.needs_update(now) ||
           vehicle_status_fi_.needs_update(now) ||
           geofencing_zones_fi_.needs_update(now);
  }

  hash_map<std::string, std::string> urls_{};

  file_info urls_fi_{};
  file_info system_information_fi_{};
  file_info vehicle_types_fi_{};
  file_info station_information_fi_{};
  file_info station_status_fi_{};
  file_info vehicle_status_fi_{};
  file_info geofencing_zones_fi_{};
};

struct compressed_bitvec {
  struct free_deleter {
    void operator()(char* p) const { std::free(p); }
  };

  std::unique_ptr<char[], free_deleter> data_{};
  int original_bytes_{};
  int compressed_bytes_{};
  std::size_t bitvec_size_{};
};

struct routing_data {
  std::vector<additional_node> additional_nodes_{};
  std::vector<geo::latlng> additional_node_coordinates_;
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>
      additional_edges_{};

  osr::bitvec<osr::node_idx_t> start_allowed_{};
  osr::bitvec<osr::node_idx_t> end_allowed_{};
  osr::bitvec<osr::node_idx_t> through_allowed_{};
  bool station_parking_{};
};

struct compressed_routing_data {
  std::vector<additional_node> additional_nodes_{};
  std::vector<geo::latlng> additional_node_coordinates_;
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>
      additional_edges_{};
  compressed_bitvec start_allowed_{};
  compressed_bitvec end_allowed_{};
  compressed_bitvec through_allowed_{};
};

struct provider_routing_data;

struct products_routing_data {
  products_routing_data(std::shared_ptr<provider_routing_data const>&& prd,
                        compressed_routing_data const& compressed);

  osr::sharing_data get_sharing_data(
      osr::node_idx_t::value_t const additional_node_offset,
      bool ignore_return_constraints) const {
    return {.start_allowed_ = &start_allowed_,
            .end_allowed_ = ignore_return_constraints ? nullptr : &end_allowed_,
            .through_allowed_ = &through_allowed_,
            .additional_node_offset_ = additional_node_offset,
            .additional_node_coordinates_ =
                compressed_.additional_node_coordinates_,
            .additional_edges_ = compressed_.additional_edges_};
  }

  std::shared_ptr<provider_routing_data const> provider_routing_data_;
  compressed_routing_data const& compressed_;

  osr::bitvec<osr::node_idx_t> start_allowed_;
  osr::bitvec<osr::node_idx_t> end_allowed_;
  osr::bitvec<osr::node_idx_t> through_allowed_;
};

using gbfs_products_idx_t =
    cista::strong<std::uint16_t, struct gbfs_products_idx_>;

struct provider_routing_data
    : std::enable_shared_from_this<provider_routing_data> {
  std::shared_ptr<products_routing_data> get_products_routing_data(
      gbfs_products_idx_t const prod_idx) const {
    return std::make_shared<products_routing_data>(
        shared_from_this(), products_.at(to_idx(prod_idx)));
  }

  std::vector<compressed_routing_data> products_;
};

struct provider_products {
  bool includes_vehicle_type(vehicle_type_idx_t const idx) const {
    return (idx == vehicle_type_idx_t::invalid() && vehicle_types_.empty()) ||
           utl::find(vehicle_types_, idx) != end(vehicle_types_);
  }

  gbfs_products_idx_t idx_{gbfs_products_idx_t::invalid()};
  std::vector<vehicle_type_idx_t> vehicle_types_;
  vehicle_form_factor form_factor_{vehicle_form_factor::kBicycle};
  propulsion_type propulsion_type_{propulsion_type::kHuman};
  return_constraint return_constraint_{};
  bool known_return_constraint_{};  // true if taken from feed, false if guessed

  bool has_vehicles_to_rent_{};
};

struct gbfs_products_ref {
  friend bool operator==(gbfs_products_ref const&,
                         gbfs_products_ref const&) = default;

  explicit operator bool() const noexcept {
    return provider_ != gbfs_provider_idx_t::invalid();
  }

  gbfs_provider_idx_t provider_{gbfs_provider_idx_t::invalid()};
  gbfs_products_idx_t products_{gbfs_products_idx_t::invalid()};
};

struct gbfs_provider {
  std::string id_;  // from config
  gbfs_provider_idx_t idx_{gbfs_provider_idx_t::invalid()};
  std::string group_id_;

  std::shared_ptr<provider_file_infos> file_infos_{};

  system_information sys_info_{};
  std::map<std::string, station> stations_{};
  vector_map<vehicle_type_idx_t, vehicle_type> vehicle_types_{};
  hash_map<std::pair<std::string, vehicle_start_type>, vehicle_type_idx_t>
      vehicle_types_map_{};
  hash_map<std::string, temp_vehicle_type> temp_vehicle_types_{};
  std::vector<vehicle_status> vehicle_status_;
  geofencing_zones geofencing_zones_{};
  geofencing_restrictions default_restrictions_{};
  std::optional<return_constraint> default_return_constraint_{};

  vector_map<gbfs_products_idx_t, provider_products> products_;
  bool has_vehicles_to_rent_{};
  geo::box bbox_{};

  std::optional<std::string> color_{};
};

struct gbfs_group {
  std::string id_;
  std::string name_;
  std::optional<std::string> color_{};

  std::vector<gbfs_provider_idx_t> providers_{};
};

struct oauth_state {
  config::gbfs::oauth_settings settings_;
  std::string access_token_{};
  std::optional<std::chrono::system_clock::time_point> expiry_{};
  unsigned expires_in_{};
};

struct provider_feed {
  bool operator==(provider_feed const& o) const { return id_ == o.id_; }
  bool operator==(std::string const& id) const { return id_ == id; }
  bool operator<(provider_feed const& o) const { return id_ < o.id_; }

  std::string id_;
  std::string url_;
  headers_t headers_{};
  std::optional<std::filesystem::path> dir_{};
  geofencing_restrictions default_restrictions_{};
  std::optional<return_constraint> default_return_constraint_{};
  std::optional<std::string> config_group_{};
  std::optional<std::string> config_color_{};
  std::shared_ptr<oauth_state> oauth_{};
  std::map<std::string, unsigned> default_ttl_{};
  std::map<std::string, unsigned> overwrite_ttl_{};
};

struct aggregated_feed {
  bool operator==(aggregated_feed const& o) const { return id_ == o.id_; }
  bool operator==(std::string const& id) const { return id_ == id; }
  bool operator<(aggregated_feed const& o) const { return id_ < o.id_; }

  bool needs_update() const {
    return !expiry_.has_value() ||
           expiry_.value() < std::chrono::system_clock::now();
  }

  std::string id_;
  std::string url_;
  headers_t headers_{};
  std::optional<std::chrono::system_clock::time_point> expiry_{};
  std::vector<provider_feed> feeds_{};
  std::shared_ptr<oauth_state> oauth_{};
  std::map<std::string, unsigned> default_ttl_{};
  std::map<std::string, unsigned> overwrite_ttl_{};
};

struct gbfs_data {
  explicit gbfs_data(std::size_t const cache_size) : cache_{cache_size} {}

  std::shared_ptr<products_routing_data> get_products_routing_data(
      osr::ways const& w, osr::lookup const& l, gbfs_products_ref);

  std::shared_ptr<std::vector<std::unique_ptr<provider_feed>>>
      standalone_feeds_{};
  std::shared_ptr<std::vector<std::unique_ptr<aggregated_feed>>>
      aggregated_feeds_{};

  vector_map<gbfs_provider_idx_t, std::unique_ptr<gbfs_provider>> providers_{};
  hash_map<std::string, gbfs_provider_idx_t> provider_by_id_{};
  point_rtree<gbfs_provider_idx_t> provider_rtree_{};
  box_rtree<gbfs_provider_idx_t> provider_zone_rtree_{};

  hash_map<std::string, gbfs_group> groups_{};

  lru_cache<gbfs_provider_idx_t, provider_routing_data> cache_;

  // used to share decompressed routing data between routing requests
  std::mutex products_routing_data_mutex_;
  hash_map<gbfs_products_ref, std::weak_ptr<products_routing_data>>
      products_routing_data_{};
};

}  // namespace motis::gbfs
