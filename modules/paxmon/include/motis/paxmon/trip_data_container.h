#pragma once

#include "utl/verify.h"

#include "motis/data.h"
#include "motis/hash_map.h"
#include "motis/vector.h"

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/trip_capacity_status.h"

namespace motis::paxmon {

struct trip_data_container {
  trip_data_index get_index(trip_idx_t const trip_idx) const {
    return mapping_.at(trip_idx);
  }

  trip_data_index find_index(trip_idx_t const trip_idx) const {
    if (auto it = mapping_.find(trip_idx); it != end(mapping_)) {
      return it->second;
    } else {
      return INVALID_TRIP_DATA_INDEX;
    }
  }

  [[nodiscard]] bool contains(trip_idx_t const trip_idx) const {
    return mapping_.find(trip_idx) != end(mapping_);
  }

  trip_data_index insert_trip(trip_idx_t const trip_idx,
                              event_node_index const enter_exit_node) {
    auto const idx = mapping_.size();
    mapping_[trip_idx] = idx;
    // init empty entries
    edges_[idx];
    canceled_nodes_[idx];
    utl::verify(enter_exit_nodes_.size() == idx,
                "insert_trip: invalid enter_exit_nodes size");
    enter_exit_nodes_.emplace_back(enter_exit_node);
    utl::verify(capacity_status_.size() == idx,
                "insert_trip: invalid capacity_status_ size");
    capacity_status_.resize(idx + 1);
    return idx;
  }

  dynamic_fws_multimap<edge_index>::const_bucket edges(trip const* trp) const {
    return edges(get_index(trp->trip_idx_));
  }

  dynamic_fws_multimap<edge_index>::mutable_bucket edges(trip const* trp) {
    return edges(get_index(trp->trip_idx_));
  }

  dynamic_fws_multimap<edge_index>::const_bucket edges(
      trip_data_index tdi) const {
    return edges_.at(tdi);
  }

  dynamic_fws_multimap<edge_index>::mutable_bucket edges(trip_data_index tdi) {
    return edges_.at(tdi);
  }

  dynamic_fws_multimap<event_node_index>::const_bucket canceled_nodes(
      trip const* trp) const {
    return canceled_nodes(get_index(trp->trip_idx_));
  }

  dynamic_fws_multimap<event_node_index>::mutable_bucket canceled_nodes(
      trip const* trp) {
    return canceled_nodes(get_index(trp->trip_idx_));
  }

  dynamic_fws_multimap<event_node_index>::const_bucket canceled_nodes(
      trip_data_index tdi) const {
    return canceled_nodes_[tdi];
  }

  dynamic_fws_multimap<event_node_index>::mutable_bucket canceled_nodes(
      trip_data_index tdi) {
    return canceled_nodes_[tdi];
  }

  event_node_index enter_exit_node(trip const* trp) const {
    return enter_exit_node(get_index(trp->trip_idx_));
  }

  event_node_index enter_exit_node(trip_data_index tdi) const {
    return enter_exit_nodes_[tdi];
  }

  trip_capacity_status const& capacity_status(trip_data_index tdi) const {
    return capacity_status_[tdi];
  }

  trip_capacity_status& capacity_status(trip_data_index tdi) {
    return capacity_status_[tdi];
  }

  std::uint32_t size() const { return mapping_.size(); }

  dynamic_fws_multimap<edge_index> edges_;
  dynamic_fws_multimap<event_node_index> canceled_nodes_;
  mcd::vector<event_node_index> enter_exit_nodes_;
  mcd::vector<trip_capacity_status> capacity_status_;
  mcd::hash_map<trip_idx_t, trip_data_index> mapping_;
};

}  // namespace motis::paxmon
