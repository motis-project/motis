#pragma once

#include <ctime>
#include <array>
#include <map>
#include <set>

#include "flatbuffers/flatbuffers.h"

#include "cista/hashing.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/common/hash_helper.h"
#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/provider.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/timezone.h"

#include "motis/loader/bitfield.h"
#include "motis/loader/loader_options.h"
#include "motis/loader/timezone_util.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace motis::loader {

struct route_section {
  route_section()
      : from_route_node_(nullptr),
        to_route_node_(nullptr),
        outgoing_route_edge_index_(-1) {}

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
    assert(from_route_node_->edges_[outgoing_route_edge_index_].type() ==
           edge::ROUTE_EDGE);
    return &from_route_node_->edges_[outgoing_route_edge_index_];
  }

  node* from_route_node_;
  node* to_route_node_;
  int outgoing_route_edge_index_;
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
  MAKE_COMPARABLE()

  Service const* service_{nullptr};
  int day_offset_{0};
};

struct services_key {
  services_key() = default;

  services_key(Service const* service, int day_idx)
      : services_({{service, 0}}), day_idx_(day_idx) {}

  services_key(std::set<service_with_day_offset> services, int day_idx)
      : services_(std::move(services)), day_idx_(day_idx) {}

  friend bool operator<(services_key const& lhs, services_key const& rhs) {
    return std::tie(lhs.services_, lhs.day_idx_) <
           std::tie(rhs.services_, rhs.day_idx_);
  }

  friend bool operator==(services_key const& lhs, services_key const& rhs) {
    return std::tie(lhs.services_, lhs.day_idx_) ==
           std::tie(rhs.services_, rhs.day_idx_);
  }

  std::set<service_with_day_offset> services_;
  int day_idx_{0};
};

template <typename T, typename... Args>
inline std::size_t push_mem(mcd::vector<mcd::unique_ptr<T>>& elements,
                            Args... args) {
  auto idx = elements.size();
  elements.emplace_back(new T{args...});
  return idx;
}

using route = mcd::vector<route_section>;
using route_lcs = mcd::vector<mcd::vector<light_connection>>;

struct graph_builder {
  graph_builder(schedule&, loader_options const&,
                unsigned progress_offset = 0U);

  full_trip_id get_full_trip_id(Service const* s, int day, int section_idx = 0);

  merged_trips_idx create_merged_trips(Service const* s, int day_idx);

  trip* register_service(Service const* s, int day_idx);

  void add_services(
      flatbuffers64::Vector<flatbuffers64::Offset<Service>> const* services);

  void index_first_route_node(route const& r);

  void add_route_services(mcd::vector<Service const*> const& services);

  void add_route_services(
      mcd::vector<std::pair<Service const*, bitfield>> const& services);

  void add_expanded_trips(route const& r);

  static int get_index(
      mcd::vector<mcd::vector<light_connection>> const& alt_route,
      mcd::vector<light_connection> const& sections);

  static void add_to_route(mcd::vector<mcd::vector<light_connection>>& route,
                           mcd::vector<light_connection> const& sections,
                           int index);

  static void add_to_routes(
      mcd::vector<mcd::vector<mcd::vector<light_connection>>>& alt_routes,
      mcd::vector<light_connection> const& sections);

  connection_info* get_or_create_connection_info(Section const* section,
                                                 int dep_day_index,
                                                 connection_info* merged_with);

  connection_info* get_or_create_connection_info(
      std::array<participant, 16> const& services, int dep_day_index);

  light_connection section_to_connection(
      merged_trips_idx trips, std::array<participant, 16> const& services,
      int day, time prev_arr, bool& adjusted);

  void connect_reverse();

  void sort_connections();
  void sort_trips();

  bitfield const& get_or_create_bitfield(
      flatbuffers64::String const* serialized_bitfield);

  void read_attributes(
      int day,
      flatbuffers64::Vector<flatbuffers64::Offset<Attribute>> const* attributes,
      mcd::vector<ptr<attribute const>>& active_attributes);

  mcd::string const* get_or_create_direction(Direction const* dir);

  provider const* get_or_create_provider(Provider const* p);

  int get_or_create_category_index(Category const* c);

  int get_or_create_track(
      int day,
      flatbuffers64::Vector<flatbuffers64::Offset<Track>> const* tracks);

  void write_trip_info(route&);

  mcd::unique_ptr<route> create_route(Route const* r, route_lcs const& lcons,
                                      unsigned route_index);

  route_section add_route_section(
      int route_index, mcd::vector<light_connection> const& connections,
      Station const* from_stop, bool from_in_allowed, bool from_out_allowed,
      Station const* to_stop, bool to_in_allowed, bool to_out_allowed,
      node* from_route_node, node* to_route_node);

  bool check_trip(trip const* trp);

  unsigned progress_offset_{0U};
  unsigned lcon_count_{0U};
  unsigned next_route_index_{0U};
  tz_cache tz_cache_;
  std::map<Category const*, int> categories_;
  std::map<std::string, int> tracks_;
  std::map<AttributeInfo const*, attribute*> attributes_;
  std::map<flatbuffers64::String const*, mcd::string const*> directions_;
  std::map<Provider const*, provider const*> providers_;
  mcd::hash_map<Station const*, station_node*> stations_;
  mcd::hash_map<flatbuffers64::String const*, bitfield> bitfields_;
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
  int first_day_{0}, last_day_{0};
  bool apply_rules_{false};
  bool expand_trips_{false};

  connection_info con_info_;
  connection con_;
  std::size_t broken_trips_{0U};
};

}  // namespace motis::loader
