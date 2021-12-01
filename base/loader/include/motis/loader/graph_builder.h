#pragma once

#include <ctime>
#include <array>
#include <map>
#include <set>

#include "flatbuffers/flatbuffers.h"

#include "cista/hashing.h"
#include "cista/reflection/comparable.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/common/hash_helper.h"
#include "motis/core/schedule/bitfield.h"
#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/provider.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/timezone.h"

#include "motis/loader/loader_options.h"
#include "motis/loader/route.h"
#include "motis/loader/timezone_util.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace motis::loader {

struct lcon_times {
  CISTA_COMPARABLE()
  uint16_t d_time_, a_time_;
};

struct route_section {
  route_section() = default;
  route_section(node* from, node* to, int edge_idx)
      : from_route_node_(from),
        to_route_node_(to),
        outgoing_route_edge_index_(edge_idx) {
    assert(from_route_node_ == nullptr || from_route_node_->is_route_node());
    assert(to_route_node_ == nullptr || to_route_node_->is_route_node());
  }

  bool is_valid() const {
    return from_route_node_ != nullptr && to_route_node_ != nullptr &&
           outgoing_route_edge_index_ != -1;
  }

  edge* get_route_edge() const {
    if (outgoing_route_edge_index_ == -1) {
      return nullptr;
    }

    assert(outgoing_route_edge_index_ >= 0);
    assert(static_cast<unsigned>(outgoing_route_edge_index_) <
           from_route_node_->edges_.size());
    assert(edge::is_route_edge(
        from_route_node_->edges_[outgoing_route_edge_index_].type()));
    return &from_route_node_->edges_[outgoing_route_edge_index_];
  }

  node* from_route_node_{nullptr};
  node* to_route_node_{nullptr};
  int outgoing_route_edge_index_{-1};
};

struct participant {
  participant() : service_(nullptr), section_idx_(0) {}

  participant(Service const* service, unsigned section_idx)
      : service_(service), section_idx_(section_idx) {}

  friend bool operator<(participant const& lhs, participant const& rhs) {
    return lhs.service_ > rhs.service_;
  }

  friend bool operator>(participant const& lhs, participant const& rhs) {
    return lhs.service_ < rhs.service_;
  }

  friend bool operator==(participant const& lhs, participant const& rhs) {
    return lhs.service_ == rhs.service_;
  }

  Service const* service_;
  unsigned section_idx_;
};

struct service_with_day_offset {
  CISTA_COMPARABLE()

  Service const* service_{nullptr};
  int day_offset_{0};
};

struct services_key {
  services_key() = default;

  explicit services_key(Service const* service) : services_({{service, 0}}) {}

  explicit services_key(std::set<service_with_day_offset> services)
      : services_(std::move(services)) {}

  friend bool operator<(services_key const& lhs, services_key const& rhs) {
    return lhs.services_ < rhs.services_;
  }

  friend bool operator==(services_key const& lhs, services_key const& rhs) {
    return lhs.services_ == rhs.services_;
  }

  std::set<service_with_day_offset> services_;
};

template <typename T, typename... Args>
inline std::size_t push_mem(mcd::vector<mcd::unique_ptr<T>>& elements,
                            Args... args) {
  auto idx = elements.size();
  elements.emplace_back(new T{args...});
  return idx;
}

using route = mcd::vector<route_section>;

struct graph_builder {
  graph_builder(schedule&, loader_options const&);

  full_trip_id get_full_trip_id(Service const* s,
                                mcd::vector<time> const& rel_utc_times,
                                size_t section_idx = 0);

  merged_trips_idx create_merged_trips(Service const*,
                                       mcd::vector<time> const& rel_utc_times);

  trip_info* register_service(Service const* s,
                              mcd::vector<time> const& rel_utc_times);

  void add_services(
      flatbuffers64::Vector<flatbuffers64::Offset<Service>> const* services);

  bool has_duplicate(Service const*, mcd::vector<light_connection> const&);

  bool are_duplicates(Service const*, mcd::vector<light_connection> const&,
                      trip_info const*);

  void index_first_route_node(route const& r);

  bool has_traffic_within_timespan(bitfield const& traffic_days,
                                   day_idx_t start_idx,
                                   day_idx_t end_idx) const;

  void add_route_services(
      mcd::vector<std::pair<Service const*, bitfield_idx_t>> const& services);

  void add_expanded_trips(route const& r);

  void dedup_bitfields();

  static int get_index(
      mcd::vector<mcd::vector<light_connection>> const& alt_route,
      mcd::vector<light_connection> const& sections);

  static void add_to_route(mcd::vector<mcd::vector<light_connection>>& route,
                           mcd::vector<light_connection> const& sections,
                           int index);

  void add_to_routes(mcd::vector<route_t>& alt_routes,
                     mcd::vector<time> const& times,
                     mcd::vector<light_connection> const& lcons);

  connection_info* get_or_create_connection_info(Section const* section,
                                                 connection_info* merged_with);

  connection_info* get_or_create_connection_info(
      std::array<participant, 16> const& services);

  light_connection section_to_connection(
      std::array<participant, 16> const& services,
      mcd::vector<time> const& relative_utc,
      mcd::vector<unsigned> const& srv_traffic_days, merged_trips_idx);

  void connect_reverse();

  void sort_connections();
  void sort_trips();

  mcd::hash_map<mcd::vector<time>, mcd::vector<unsigned>> service_times_to_utc(
      bitfield const& traffic_days, day_idx_t start_idx, day_idx_t end_idx,
      Service const* s);

  bitfield_idx_t store_bitfield(bitfield const&);
  bitfield_idx_t get_or_create_bitfield(
      flatbuffers64::String const* serialized_bitfield, day_idx_t offset = 0);

  mcd::string const* get_or_create_direction(Direction const* dir);
  mcd::string const* get_or_create_string(flatbuffers64::String const* str);

  provider const* get_or_create_provider(Provider const* p);

  int get_or_create_category_index(Category const* c);

  uint32_t get_or_create_track(
      flatbuffers64::Vector<flatbuffers64::Offset<Track>> const* tracks,
      day_idx_t offset);

  void write_trip_edges(route const& r);

  mcd::unique_ptr<route> create_route(Route const* r, route_t const& lcons,
                                      unsigned route_index);

  route_section add_route_section(
      int route_index, mcd::vector<light_connection> const& connections,
      Station const* from_stop, bool from_in_allowed, bool from_out_allowed,
      Station const* to_stop, bool to_in_allowed, bool to_out_allowed,
      node* from_route_node, node* to_route_node, size_t route_traffic_days);

  bool check_trip(trip_info const* trp);

  bool skip_station(Station const* station) const;
  bool skip_route(Route const* route) const;

  unsigned lcon_count_{0U};
  unsigned next_route_index_{0U};
  tz_cache tz_cache_;
  std::map<Category const*, int> categories_;
  std::map<std::string, int> tracks_;
  std::map<AttributeInfo const*, attribute*> attributes_;
  std::map<flatbuffers64::String const*, mcd::string const*> strings_;
  std::map<Provider const*, provider const*> providers_;
  mcd::hash_map<Station const*, station_node*> stations_;
  mcd::hash_map<mcd::pair<flatbuffers64::String const*, day_idx_t /* offset */>,
                bitfield>
      bitfields_;
  mcd::hash_set<connection_info*,
                deep_ptr_hash<cista::hashing<connection_info>, connection_info>,
                deep_ptr_eq<connection_info>>
      con_infos_;
  mcd::hash_set<connection*,
                deep_ptr_hash<cista::hashing<connection>, connection>,
                deep_ptr_eq<connection>>
      connections_;
  mcd::hash_map<flatbuffers64::String const*, mcd::string*> filenames_;
  schedule& sched_;
  day_idx_t first_day_{0}, last_day_{0};
  bool apply_rules_{false};
  bool expand_trips_{false};
  bool no_local_transport_{false};

  connection_info con_info_;
  connection con_;
  std::size_t broken_trips_{0U};
};

}  // namespace motis::loader
