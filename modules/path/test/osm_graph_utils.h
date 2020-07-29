#pragma once

#include "motis/path/prepare/osm/osm_graph.h"

namespace motis::path {

inline void add_node(osm_graph& graph, size_t idx, bool terminal = false) {
  utl::verify(graph.nodes_.size() == idx, "add_node: node idx mismatch");
  graph.nodes_.emplace_back(
      std::make_unique<osm_node>(idx, 0, 0, geo::latlng{}));
  if (terminal) {
    graph.node_station_links_.emplace_back("foo", idx, 42);
  }
}

inline void add_edge(osm_graph& graph, size_t from, size_t to, size_t dist) {
  auto* from_node = graph.nodes_[from].get();
  auto* to_node = graph.nodes_[to].get();
  from_node->edges_.emplace_back(0, true, dist, from_node, to_node);
}

inline void add_edge2(osm_graph& graph, size_t from, size_t to, size_t dist) {
  add_edge(graph, from, to, dist);
  add_edge(graph, to, from, dist);
}

inline void set_single_component(osm_graph& graph) {
  graph.components_ = 1ULL;
  graph.component_offsets_ = {{0ULL, graph.nodes_.size()}};
}

}  // namespace motis::path
