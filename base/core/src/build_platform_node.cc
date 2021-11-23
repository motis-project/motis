#include "motis/core/schedule/build_platform_node.h"

namespace motis {

node* get_or_add_platform_node(schedule& sched, station_node* station_node,
                               uint16_t platform) {
  if (platform >= station_node->platform_nodes_.size()) {
    station_node->platform_nodes_.resize(platform + 1);
  }

  auto platform_node = station_node->platform_nodes_[platform];
  if (platform_node == nullptr) {
    platform_node =
        station_node->child_nodes_
            .emplace_back(mcd::make_unique<node>(make_node(
                node_type::PLATFORM_NODE, station_node, sched.next_node_id_++)))
            .get();
    station_node->platform_nodes_[platform] = platform_node;
  }
  return platform_node;
}

node* add_platform_enter_edge(schedule& sched, node* route_node,
                              station_node* station_node,
                              int32_t platform_transfer_time,
                              uint16_t platform) {
  auto const pn = get_or_add_platform_node(sched, station_node, platform);
  pn->edges_.push_back(
      make_enter_edge(pn, route_node, platform_transfer_time, true));
  return pn;
}

node* add_platform_exit_edge(schedule& sched, node* route_node,
                             station_node* station_node,
                             int32_t platform_transfer_time,
                             uint16_t platform) {
  auto const pn = get_or_add_platform_node(sched, station_node, platform);
  route_node->edges_.push_back(
      make_exit_edge(route_node, pn, platform_transfer_time, true));
  return pn;
}

}  // namespace motis
