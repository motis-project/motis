#include "motis/core/access/edge_access.h"

#include <cassert>
#include <iterator>

namespace motis {

edge const* get_route_edge(node const* route_node,
                           generic_light_connection const& lcon,
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

time get_rt_time(edge const* route_edge, std::size_t const lcon_index,
                 event_type const ev_type) {
  return route_edge->rt_lcons().at(lcon_index).event_time(ev_type);
}

}  // namespace motis
