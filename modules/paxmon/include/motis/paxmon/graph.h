#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/pax_connection_info.h"

namespace motis::paxmon {

struct edge;
struct graph;

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
  inline station const& get_station(schedule const& sched) {
    return *sched.stations_[station_idx()];
  }

  time time_{INVALID_TIME};
  time schedule_time_{INVALID_TIME};
  event_type type_{event_type::ARR};
  bool valid_{true};
  std::uint32_t station_{0};

  std::vector<std::unique_ptr<edge>> out_edges_;
  std::vector<edge*> in_edges_;
};

enum class edge_type : std::uint8_t { TRIP, INTERCHANGE, WAIT };

inline std::ostream& operator<<(std::ostream& out, edge_type const et) {
  switch (et) {
    case edge_type::TRIP: return out << "TRIP";
    case edge_type::INTERCHANGE: return out << "INTERCHANGE";
    case edge_type::WAIT: return out << "WAIT";
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
  inline bool is_trip() const { return type() != edge_type::INTERCHANGE; }
  inline bool is_interchange() const {
    return type() == edge_type::INTERCHANGE;
  }

  inline std::uint16_t passengers_over_capacity() const {
    return passengers_ > capacity_ ? passengers_ - capacity_ : 0U;
  }

  inline event_node* from(graph const&) const { return from_; }
  inline event_node* to(graph const&) const { return to_; }

  inline edge_type type() const { return type_; }
  inline trip const* get_trip() const { return trip_; }
  inline duration transfer_time() const { return transfer_time_; }
  inline std::uint64_t capacity() const { return capacity_; }
  inline std::uint64_t passengers() const { return passengers_; }
  inline bool is_broken() const { return broken_; }
  inline pax_connection_info const& get_pax_connection_info() const {
    return pax_connection_info_;
  }

  event_node* from_{};
  event_node* to_{};
  edge_type type_{};
  bool broken_{false};
  duration transfer_time_{};
  std::uint16_t capacity_{};
  std::uint16_t passengers_{};
  struct trip const* trip_{};
  struct pax_connection_info pax_connection_info_;
};

struct trip_data {
  std::vector<edge*> edges_;
  std::vector<event_node*> canceled_nodes_;
  event_node enter_exit_node_;
};

struct graph {
  std::vector<std::unique_ptr<event_node>> nodes_;
  mcd::hash_map<extern_trip, std::unique_ptr<trip_data>> trip_data_;
  std::vector<std::unique_ptr<passenger_group>> passenger_groups_;
};

}  // namespace motis::paxmon
