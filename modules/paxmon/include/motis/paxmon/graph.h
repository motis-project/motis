#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "utl/enumerate.h"

#include "motis/data.h"
#include "motis/vector.h"

#include "motis/core/common/fws_graph.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/schedule/trip_idx.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/paxmon/capacity_data.h"
#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/passenger_group_container.h"
#include "motis/paxmon/pax_connection_info.h"
#include "motis/paxmon/trip_data_container.h"

namespace motis::paxmon {

struct edge;
struct graph;

struct event_node {
  using mutable_outgoing_edge_bucket =
      typename fws_graph<event_node, edge>::mutable_outgoing_edge_bucket;
  using const_outgoing_edge_bucket =
      typename fws_graph<event_node, edge>::const_outgoing_edge_bucket;

  using mutable_incoming_edge_bucket =
      fws_graph<event_node, edge>::mutable_incoming_edge_bucket;
  using const_incoming_edge_bucket =
      fws_graph<event_node, edge>::const_incoming_edge_bucket;

  inline bool is_valid() const { return valid_; }
  inline bool is_canceled() const { return !valid_; }

  const_outgoing_edge_bucket outgoing_edges(graph const& g) const;
  mutable_outgoing_edge_bucket outgoing_edges(graph& g) const;

  const_incoming_edge_bucket incoming_edges(graph const& g) const;

  inline time current_time() const { return time_; }
  inline time schedule_time() const { return schedule_time_; }
  inline event_type type() const { return type_; }
  inline std::uint32_t station_idx() const { return station_; }
  inline station const& get_station(schedule const& sched) const {
    return *sched.stations_[station_idx()];
  }

  inline event_node_index index(graph const&) const { return index_; }

  event_node_index index_{};
  time time_{INVALID_TIME};
  time schedule_time_{INVALID_TIME};
  event_type type_{event_type::ARR};
  bool valid_{true};
  std::uint32_t station_{0};
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

  event_node const* from(graph const&) const;
  event_node* from(graph&) const;

  event_node const* to(graph const&) const;
  event_node* to(graph&) const;

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

  inline pax_connection_info& get_pax_connection_info() {
    return pax_connection_info_;
  }

  event_node_index from_{};
  event_node_index to_{};
  edge_type type_{};
  bool broken_{false};
  duration transfer_time_{};
  std::uint16_t encoded_capacity_{};
  service_class clasz_{service_class::OTHER};
  merged_trips_idx trips_{};
  struct pax_connection_info pax_connection_info_;
};

struct graph {
  fws_graph<event_node, edge> graph_;
  trip_data_container trip_data_;
  passenger_group_container passenger_groups_;
};

}  // namespace motis::paxmon
