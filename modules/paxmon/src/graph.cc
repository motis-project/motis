#include "motis/paxmon/graph.h"

namespace motis::paxmon {

event_node::const_outgoing_edge_bucket event_node::outgoing_edges(
    graph const& g) const {
  return g.graph_.outgoing_edges(index_);
}

event_node::mutable_outgoing_edge_bucket event_node::outgoing_edges(
    graph& g) const {
  return g.graph_.outgoing_edges(index_);
}

event_node::const_incoming_edge_bucket event_node::incoming_edges(
    graph const& g) const {
  return g.graph_.incoming_edges(index_);
}

event_node const* edge::from(graph const& g) const {
  return &g.graph_.nodes_[from_];
}

event_node* edge::from(graph& g) const { return &g.graph_.nodes_[from_]; }

event_node const* edge::to(graph const& g) const {
  return &g.graph_.nodes_[to_];
}

event_node* edge::to(graph& g) const { return &g.graph_.nodes_[to_]; }

}  // namespace motis::paxmon
