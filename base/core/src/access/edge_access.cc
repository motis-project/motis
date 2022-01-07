#include "motis/core/access/edge_access.h"

#include <cassert>
#include <iterator>

namespace motis {

edge const* get_route_edge(node const* route_node,
                           rt_light_connection const* lcon,
                           event_type const ev_type) {
  if (ev_type == event_type::DEP) {
    for (auto const& e : route_node->edges_) {
      if (e.contains(lcon)) {
        return &e;
      }
    }
  } else {
    for (auto const& e : route_node->incoming_edges_) {
      if (e->contains(lcon)) {
        return e;
      }
    }
  }
  throw std::runtime_error("get_route_edge(): light connection not found");
}

node const* get_route_node(edge const& e, event_type const ev_type) {
  return ev_type == event_type::DEP ? e.from() : e.to();
}

time get_time(rt_light_connection const* lcon, event_type const ev_type) {
  return lcon->event_time(ev_type);
}

lcon_idx_t get_lcon_index(edge const* route_edge,
                          rt_light_connection const* lcon) {
  auto const& lcons = route_edge->rt_lcons();
  if (lcon < begin(lcons) || lcon >= end(lcons)) {
    throw std::runtime_error("get_lcon_index(): light connection not found");
  }
  return static_cast<lcon_idx_t>(std::distance(begin(lcons), lcon));
}

}  // namespace motis
