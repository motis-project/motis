#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "ctx/res_id_t.h"

#include "utl/enumerate.h"

#include "motis/data.h"
#include "motis/vector.h"

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/core/common/fws_graph.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/schedule/trip_idx.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/module/global_res_ids.h"

#include "motis/paxmon/capacity_data.h"
#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/passenger_group_container.h"
#include "motis/paxmon/pci_container.h"
#include "motis/paxmon/rt_update_context.h"
#include "motis/paxmon/statistics.h"
#include "motis/paxmon/trip_data_container.h"
#include "motis/paxmon/update_tracker.h"

namespace motis::paxmon {

struct edge;
struct universe;

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
  inline bool is_enter_exit_node() const { return station_ == 0; }

  const_outgoing_edge_bucket outgoing_edges(universe const&) const;
  mutable_outgoing_edge_bucket outgoing_edges(universe&) const;

  const_incoming_edge_bucket incoming_edges(universe const&) const;

  inline time current_time() const { return time_; }
  inline time schedule_time() const { return schedule_time_; }
  inline event_type type() const { return type_; }
  inline std::uint32_t station_idx() const { return station_; }
  inline station const& get_station(schedule const& sched) const {
    return *sched.stations_[station_idx()];
  }

  inline event_node_index index(universe const&) const { return index_; }

  event_node_index index_{};
  time time_{INVALID_TIME};
  time schedule_time_{INVALID_TIME};
  event_type type_{event_type::ARR};
  bool valid_{true};
  std::uint32_t station_{0};
};

enum class edge_type : std::uint8_t {
  TRIP,
  INTERCHANGE,
  WAIT,
  THROUGH,
  DISABLED
};

inline std::ostream& operator<<(std::ostream& out, edge_type const et) {
  switch (et) {
    case edge_type::TRIP: return out << "TRIP";
    case edge_type::INTERCHANGE: return out << "INTERCHANGE";
    case edge_type::WAIT: return out << "WAIT";
    case edge_type::THROUGH: return out << "THROUGH";
    case edge_type::DISABLED: return out << "DISABLED";
  }
  return out;
}

struct edge {
  inline bool is_valid(universe const& u) const {
    return !is_disabled() && from(u)->is_valid() && to(u)->is_valid();
  }

  inline bool is_canceled(universe const& u) const {
    return is_disabled() || from(u)->is_canceled() || to(u)->is_canceled();
  }

  inline bool is_trip() const { return type() == edge_type::TRIP; }

  inline bool is_interchange() const {
    return type() == edge_type::INTERCHANGE;
  }

  inline bool is_wait() const { return type() == edge_type::WAIT; }

  inline bool is_disabled() const { return type() == edge_type::DISABLED; }

  event_node const* from(universe const&) const;
  event_node* from(universe&) const;

  event_node const* to(universe const&) const;
  event_node* to(universe&) const;

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

  event_node_index from_{};
  event_node_index to_{};
  edge_type type_{};
  bool broken_{false};
  duration transfer_time_{};
  std::uint16_t encoded_capacity_{};
  service_class clasz_{service_class::OTHER};
  merged_trips_idx trips_{};
  pci_index pci_{};
};

using universe_id = std::uint32_t;

struct universe {
  passenger_group const* get_passenger_group(passenger_group_index id) const;

  bool uses_default_schedule() const {
    return schedule_res_id_ ==
           motis::module::to_res_id(motis::module::global_res_id::SCHEDULE);
  }

  universe_id id_{};
  ctx::res_id_t schedule_res_id_{};

  fws_graph<event_node, edge> graph_;
  trip_data_container trip_data_;
  passenger_group_container passenger_groups_;
  pci_container pax_connection_info_;
  dynamic_fws_multimap<edge_index> interchanges_at_station_;

  rt_update_context rt_update_ctx_;
  system_statistics system_stats_;
  tick_statistics tick_stats_;
  tick_statistics last_tick_stats_;
  update_tracker update_tracker_;
};

}  // namespace motis::paxmon
