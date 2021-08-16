#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "utl/enumerate.h"

#include "motis/data.h"
#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/schedule/trip_idx.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/paxmon/allocator.h"
#include "motis/paxmon/capacity_data.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/pax_connection_info.h"

namespace motis::paxmon {

struct edge;
struct edge_idx;
struct graph;

using event_node_idx = std::uint32_t;

edge* get_edge(graph const& g, edge_idx const& ei);

struct edge_idx {
  inline edge* get(graph const& g) const { return get_edge(g, *this); }

  event_node_idx node_{};
  std::uint32_t out_edge_idx_{};
};

struct event_node {
  inline bool is_valid() const { return valid_; }
  inline bool is_canceled() const { return !valid_; }

  inline std::vector<std::unique_ptr<edge>> const& outgoing_edges(
      graph const&) const {
    return out_edges_;
  }

  inline std::vector<edge*> const& incoming_edges(graph const&) const {
    return in_edges_;
  }

  inline time current_time() const { return time_; }
  inline time schedule_time() const { return schedule_time_; }
  inline event_type type() const { return type_; }
  inline std::uint32_t station_idx() const { return station_; }
  inline station const& get_station(schedule const& sched) const {
    return *sched.stations_[station_idx()];
  }

  inline std::uint32_t index(graph const&) const { return index_; }

  event_node_idx index_{};
  time time_{INVALID_TIME};
  time schedule_time_{INVALID_TIME};
  event_type type_{event_type::ARR};
  bool valid_{true};
  std::uint32_t station_{0};

  std::vector<std::unique_ptr<edge>> out_edges_;
  std::vector<edge*> in_edges_;
};

enum class edge_type : std::uint8_t { TRIP, INTERCHANGE, WAIT, THROUGH };

inline std::ostream& operator<<(std::ostream& out, edge_type const et) {
  switch (et) {
    case edge_type::TRIP: return out << "TRIP";
    case edge_type::INTERCHANGE: return out << "INTERCHANGE";
    case edge_type::WAIT: return out << "WAIT";
    case edge_type::THROUGH: return out << "THROUGH";
  }
  return out;
}

struct edge {
  inline bool is_valid(graph const& g) const {
    return from(g)->is_valid() && to(g)->is_valid();
  }

  inline bool is_canceled(graph const& g) const {
    return from(g)->is_canceled() || to(g)->is_canceled();
  }

  inline bool is_trip() const { return type() == edge_type::TRIP; }

  inline bool is_interchange() const {
    return type() == edge_type::INTERCHANGE;
  }

  inline bool is_wait() const { return type() == edge_type::WAIT; }

  inline event_node* from(graph const&) const { return from_; }
  inline event_node* to(graph const&) const { return to_; }

  inline edge_type type() const { return type_; }

  inline merged_trips_idx get_merged_trips_idx() const { return trips_; }

  inline mcd::vector<ptr<trip>> const& get_trips(schedule const& sched) const {
    return *sched.merged_trips_.at(trips_);
  }

  inline duration transfer_time() const { return transfer_time_; }

  inline std::uint16_t capacity() const {
    return get_capacity(encoded_capacity_);
  }

  inline capacity_source get_capacity_source() const {
    return ::motis::paxmon::get_capacity_source(encoded_capacity_);
  }

  inline bool has_unlimited_capacity() const {
    return encoded_capacity_ == UNLIMITED_ENCODED_CAPACITY;
  }

  inline bool has_unknown_capacity() const {
    return encoded_capacity_ == UNKNOWN_ENCODED_CAPACITY;
  }

  inline bool has_capacity() const {
    return !has_unknown_capacity() && !has_unlimited_capacity();
  }

  inline bool is_broken() const { return broken_; }

  inline pax_connection_info const& get_pax_connection_info() const {
    return pax_connection_info_;
  }

  event_node* from_{};
  event_node* to_{};
  edge_type type_{};
  bool broken_{false};
  duration transfer_time_{};
  std::uint16_t encoded_capacity_{};
  service_class clasz_{service_class::OTHER};
  merged_trips_idx trips_{};
  struct pax_connection_info pax_connection_info_;
};

struct trip_data {
  std::vector<edge_idx> edges_;
  std::vector<event_node_idx> canceled_nodes_;
  event_node_idx enter_exit_node_{};
};

struct graph {
  inline passenger_group* add_group(passenger_group&& pg) {
    auto const id = static_cast<std::uint64_t>(passenger_groups_.size());
    auto ptr = passenger_groups_.emplace_back(
        passenger_group_allocator_.create(std::move(pg)));
    ptr->id_ = id;
    groups_by_source_[ptr->source_].emplace_back(id);
    return ptr;
  }

  std::vector<std::unique_ptr<event_node>> nodes_;
  mcd::hash_map<trip const*, std::unique_ptr<trip_data>> trip_data_;
  std::vector<passenger_group*> passenger_groups_;
  allocator<passenger_group> passenger_group_allocator_;
  mcd::hash_map<data_source, mcd::vector<std::uint64_t>> groups_by_source_;
};

inline edge* get_edge(graph const& g, edge_idx const& ei) {
  return g.nodes_.at(ei.node_)->outgoing_edges(g).at(ei.out_edge_idx_).get();
}

inline edge_idx get_edge_idx(graph const& g, edge const* e) {
  auto const from = e->from(g);
  auto const node_idx = from->index(g);
  for (auto const& [i, ep] : utl::enumerate(from->outgoing_edges(g))) {
    if (ep.get() == e) {
      return edge_idx{node_idx, static_cast<std::uint32_t>(i)};
    }
  }
  throw std::runtime_error{"get_edge_idx: edge not found"};
}

}  // namespace motis::paxmon
