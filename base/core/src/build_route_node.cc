#include "motis/core/schedule/build_route_node.h"

#include "motis/core/schedule/nodes.h"

namespace motis {

node* build_route_node(int route_index, int node_id, station_node* station_node,
                       int transfer_time, bool in_allowed, bool out_allowed) {
  auto route_node =
      station_node->child_nodes_
          .emplace_back(mcd::make_unique<node>(make_node(
              node_type::ROUTE_NODE, station_node, node_id, route_index)))
          .get();

  if (!in_allowed) {
    station_node->edges_.push_back(make_invalid_edge(station_node, route_node));
  } else {
    station_node->edges_.push_back(
        make_enter_edge(station_node, route_node, transfer_time, true));
  }

  if (!out_allowed) {
    route_node->edges_.push_back(make_invalid_edge(route_node, station_node));
  } else {
    route_node->edges_.push_back(
        make_exit_edge(route_node, station_node, transfer_time, true));
  }

  if (station_node->foot_node_) {
    if (out_allowed) {
      route_node->edges_.push_back(make_after_train_fwd_edge(
          route_node, station_node->foot_node_.get(), 0, true));
    }
    if (in_allowed) {
      station_node->foot_node_->edges_.emplace_back(make_after_train_bwd_edge(
          station_node->foot_node_.get(), route_node, 0, true));
    }
  }

  return route_node;
}

}  // namespace motis
