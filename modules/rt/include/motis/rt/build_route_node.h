#pragma once

#include "motis/core/schedule/build_route_node.h"
#include "motis/core/schedule/schedule.h"

#include "motis/rt/incoming_edges.h"
#include "motis/rt/update_constant_graph.h"

namespace motis::rt {

inline node* build_route_node(schedule& sched, int route_index, int node_id,
                              station_node* station_node, int transfer_time,
                              bool in_allowed, bool out_allowed,
                              std::vector<incoming_edge_patch>& incoming) {
  auto const n = build_route_node(route_index, node_id, station_node,
                                  transfer_time, in_allowed, out_allowed);
  add_outgoing_edge(&station_node->edges_.back(), incoming);
  if (station_node->foot_node_ && in_allowed) {
    add_outgoing_edge(&station_node->foot_node_->edges_.back(), incoming);
  }
  add_outgoing_edges(n, incoming);
  constant_graph_add_route_node(sched, route_index, station_node, in_allowed,
                                out_allowed);
  return n;
}

}  // namespace motis::rt
