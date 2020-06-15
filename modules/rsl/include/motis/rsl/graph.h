#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/journey/extern_trip.h"

#include "motis/rsl/passenger_group.h"
#include "motis/rsl/rsl_connection_info.h"

namespace motis::rsl {

struct edge;

struct event_node {
  inline bool is_valid() const { return valid_; }
  inline bool is_canceled() const { return !valid_; }

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
  inline bool is_valid() const { return from_->is_valid() && to_->is_valid(); }
  inline bool is_canceled() const {
    return from_->is_canceled() || to_->is_canceled();
  }

  inline std::uint16_t passengers_over_capacity() const {
    return passengers_ > capacity_ ? passengers_ - capacity_ : 0U;
  }

  event_node* from_{};
  event_node* to_{};
  edge_type type_{};
  trip const* trip_{};  // TODO(pablo): union trip/transfer_time
  duration transfer_time_{};
  std::uint16_t capacity_{};
  std::uint16_t passengers_{};
  bool broken_{false};
  rsl_connection_info rsl_connection_info_;
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

}  // namespace motis::rsl
