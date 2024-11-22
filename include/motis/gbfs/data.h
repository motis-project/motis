#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "tg.h"

#include "cista/hash.h"
#include "cista/strong.h"

#include "geo/latlng.h"

#include "osr/routing/additional_edge.h"
#include "osr/routing/sharing_data.h"
#include "osr/types.h"

#include "motis/config.h"
#include "motis/point_rtree.h"
#include "motis/types.h"

#include "motis/gbfs/lru_cache.h"

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

  std::shared_ptr<tg_geom> station_area_{};
};

struct station_status {
  unsigned num_vehicles_available_{};
  hash_map<std::string, unsigned> vehicle_types_available_{};
  hash_map<std::string, unsigned> vehicle_docks_available_{};
  bool is_renting_{true};
  bool is_returning_{true};
};

struct station {
  bool operator==(station const& o) const { return info_.id_ == o.info_.id_; }
  auto operator<=>(station const& o) const { return info_.id_ <=> o.info_.id_; }

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
      : geom_{geom, tg_geom_deleter{}},
        rules_{std::move(rules)},
        clockwise_{geom_ && tg_geom_num_polys(geom_.get()) > 0
                       ? tg_poly_clockwise(tg_geom_poly_at(geom_.get(), 0))
                       : true},
        name_{std::move(name)} {}

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
      std::string const& vehicle_type_id,
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
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>
      additional_edges_{};

  osr::bitvec<osr::node_idx_t> start_allowed_{};
  osr::bitvec<osr::node_idx_t> end_allowed_{};
  osr::bitvec<osr::node_idx_t> through_allowed_{};
  bool station_parking_{};
};

struct compressed_routing_data {
  std::vector<additional_node> additional_nodes_{};
  osr::hash_map<osr::node_idx_t, std::vector<osr::additional_edge>>
      additional_edges_{};

  compressed_bitvec start_allowed_{};
  compressed_bitvec end_allowed_{};
  compressed_bitvec through_allowed_{};
};

struct provider_routing_data;

struct segment_routing_data {
  segment_routing_data(std::shared_ptr<provider_routing_data const>&& prd,
                       compressed_routing_data const& compressed);

  osr::sharing_data get_sharing_data(
      osr::node_idx_t::value_t const additional_node_offset) const {
    return {.start_allowed_ = start_allowed_,
            .end_allowed_ = end_allowed_,
            .through_allowed_ = through_allowed_,
            .additional_node_offset_ = additional_node_offset,
            .additional_edges_ = compressed_.additional_edges_};
  }

  std::shared_ptr<provider_routing_data const> provider_routing_data_;
  compressed_routing_data const& compressed_;

  osr::bitvec<osr::node_idx_t> start_allowed_;
  osr::bitvec<osr::node_idx_t> end_allowed_;
  osr::bitvec<osr::node_idx_t> through_allowed_;
};

using gbfs_segment_idx_t = cista::strong<std::size_t, struct gbfs_segment_idx_>;

struct provider_routing_data
    : std::enable_shared_from_this<provider_routing_data> {
  std::shared_ptr<segment_routing_data> get_segment_routing_data(
      gbfs_segment_idx_t const seg_idx) const {
    return std::make_shared<segment_routing_data>(
        shared_from_this(), segments_.at(to_idx(seg_idx)));
  }

  std::vector<compressed_routing_data> segments_;
};

struct provider_segment {
  bool includes_vehicle_type(std::string const& id) const {
    return (id.empty() && vehicle_types_.empty()) ||
           std::find(begin(vehicle_types_), end(vehicle_types_), id) !=
               end(vehicle_types_);
  }

  gbfs_segment_idx_t idx_{gbfs_segment_idx_t::invalid()};
  std::vector<std::string> vehicle_types_;
  vehicle_form_factor form_factor_{vehicle_form_factor::kBicycle};

  bool has_vehicles_to_rent_{};
};

struct gbfs_segment_ref {
  friend bool operator==(gbfs_segment_ref const&,
                         gbfs_segment_ref const&) = default;

  explicit operator bool() const noexcept {
    return provider_ != gbfs_provider_idx_t::invalid();
  }

  gbfs_provider_idx_t provider_{gbfs_provider_idx_t::invalid()};
  gbfs_segment_idx_t segment_{gbfs_segment_idx_t::invalid()};
};

struct gbfs_provider {
  std::string id_;  // from config
  gbfs_provider_idx_t idx_{};

  std::shared_ptr<provider_file_infos> file_infos_{};

  system_information sys_info_{};
  std::map<std::string, station> stations_{};
  hash_map<std::string, vehicle_type> vehicle_types_{};
  std::vector<vehicle_status> vehicle_status_;
  geofencing_zones geofencing_zones_{};
  geofencing_restrictions default_restrictions_{};

  vector_map<gbfs_segment_idx_t, provider_segment> segments_;
  bool has_vehicles_to_rent_{};
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
  std::vector<provider_feed> feeds_;
};

struct gbfs_data {
  explicit gbfs_data(std::size_t const cache_size) : cache_{cache_size} {}

  std::shared_ptr<std::vector<std::unique_ptr<provider_feed>>>
      standalone_feeds_{};
  std::shared_ptr<std::vector<std::unique_ptr<aggregated_feed>>>
      aggregated_feeds_{};

  vector_map<gbfs_provider_idx_t, std::unique_ptr<gbfs_provider>> providers_{};
  hash_map<std::string, gbfs_provider_idx_t> provider_by_id_{};
  point_rtree<gbfs_provider_idx_t> provider_rtree_{};

  lru_cache<gbfs_provider_idx_t, provider_routing_data> cache_;
};

}  // namespace motis::gbfs
