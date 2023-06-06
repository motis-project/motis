#include "motis/paxmon/universe.h"

namespace motis::paxmon {

event_node::const_outgoing_edge_bucket event_node::outgoing_edges(
    universe const& u) const {
  return u.graph_.outgoing_edges(index_);
}

event_node::mutable_outgoing_edge_bucket event_node::outgoing_edges(
    universe& u) const {
  return u.graph_.outgoing_edges(index_);
}

event_node::const_incoming_edge_bucket event_node::incoming_edges(
    universe const& u) const {
  return u.graph_.incoming_edges(index_);
}

event_node const* edge::from(universe const& u) const {
  return &u.graph_.nodes_[from_];
}

event_node* edge::from(universe& u) const { return &u.graph_.nodes_[from_]; }

event_node const* edge::to(universe const& u) const {
  return &u.graph_.nodes_[to_];
}

event_node* edge::to(universe& u) const { return &u.graph_.nodes_[to_]; }

bool edge::is_valid(universe const& u) const {
  return !is_disabled() && from(u)->is_valid() && to(u)->is_valid();
}

bool edge::is_canceled(universe const& u) const {
  return is_disabled() || from(u)->is_canceled() || to(u)->is_canceled();
}

}  // namespace motis::paxmon
