#pragma once

#include <cstdint>
#include <limits>

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"

#include "motis/paxmon/edge_type.h"
#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

struct node_log_entry {
  unixtime system_time_{INVALID_TIME};
  time node_time_{INVALID_TIME};
  bool valid_{};
};

struct edge_log_entry {
  static std::int16_t const INVALID_TRANSFER_TIME =
      std::numeric_limits<std::int16_t>::min();

  inline bool has_available_transfer_time() const {
    return available_transfer_time_ != INVALID_TRANSFER_TIME;
  }

  unixtime system_time_{INVALID_TIME};
  duration required_transfer_time_{};
  std::int16_t available_transfer_time_{
      std::numeric_limits<std::int16_t>::min()};
  edge_type edge_type_{};
  bool broken_{};
};

enum class pci_log_action_t : std::uint8_t {
  ROUTE_ADDED,
  ROUTE_REMOVED,
  BROKEN_ROUTE_ADDED,
  BROKEN_ROUTE_REMOVED
};

enum class pci_log_reason_t : std::uint8_t {
  UNKNOWN,
  API,
  TRIP_REROUTE,
  UPDATE_LOAD
};

struct pci_log_entry {
  unixtime system_time_{INVALID_TIME};
  pci_log_action_t action_{};
  pci_log_reason_t reason_{};
  passenger_group_with_route pgwr_{};
};

struct graph_log {
  dynamic_fws_multimap<node_log_entry> node_log_;  // index: event_node_index
  dynamic_fws_multimap<edge_log_entry> edge_log_;  // index: pci_index
  dynamic_fws_multimap<pci_log_entry> pci_log_;  // index: pci_index
  bool enabled_{};
};

}  // namespace motis::paxmon
