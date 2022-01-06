#include "motis/core/access/edge_access.h"

#include <cassert>
#include <iterator>

namespace motis {

edge const* get_route_edge(node const* route_node,
                           rt_light_connection const* lcon,
                           event_type const ev_type) {
  if (ev_type == event_type::DEP) {
    for (auto const& e : route_node->edges_) {
      if (e.type() == edge_type::RT_ROUTE_EDGE &&
          mcd::get<edge::rt_route_edge>(e.data_).conns_.contains(lcon)) {
        return &e;
      }
    }
  } else {
    for (auto const& e : route_node->incoming_edges_) {
      if (e->is_route_edge() &&
          mcd::get<edge::rt_route_edge>(e->data_).conns_.contains(lcon)) {
        return e;
      }
    }
  }
  throw std::runtime_error("get_route_edge(): light connection not found");
}

node const* get_route_node(edge const& e, event_type const ev_type) {
  return ev_type == event_type::DEP ? e.from() : e.to();
}

rt_light_connection const& get_lcon(edge const* route_edge,
                                    size_t const index) {
  assert(route_edge->is_route_edge());
  assert(index <
         mcd::get<edge::rt_route_edge>(route_edge->data_).conns_.size());
  return mcd::get<edge::rt_route_edge>(route_edge->data_).conns_[index];
}

time get_time(rt_light_connection const* lcon, event_type const ev_type) {
  return lcon->event_time(ev_type);
}

time get_time(edge const* route_edge, std::size_t const lcon_index,
              event_type const ev_type) {
  return get_time(&get_lcon(route_edge, lcon_index), ev_type);
}

lcon_idx_t get_lcon_index(edge const* route_edge,
                          rt_light_connection const* lcon) {
  auto const& lcons = mcd::get<edge::rt_route_edge>(route_edge->data_).conns_;
  if (lcon < begin(lcons) || lcon >= end(lcons)) {
    throw std::runtime_error("get_lcon_index(): light connection not found");
  }
  return static_cast<lcon_idx_t>(std::distance(begin(lcons), lcon));
}

}  // namespace motis
