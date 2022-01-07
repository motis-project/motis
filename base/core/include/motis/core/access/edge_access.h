#pragma once

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"

namespace motis {

/*
TODO
template <typename F>
void foreach_departure_in(edge const& edge, time const begin, time const end,
                          F&& fun) {
  if (edge.type() != edge::ROUTE_EDGE) {
    return;
  }

  auto const& conns = edge.static_lcons();
  auto const start = std::lower_bound(std::begin(conns), std::end(conns),
                                      light_connection(begin));
  for (auto it = start; it != std::end(conns) && it->d_time_ < end; ++it) {
    fun(it);
  }
}

template <typename F>
void foreach_arrival_in(edge const& edge, time const begin, time const end,
                        F&& fun) {
  if (edge.type() != edge::ROUTE_EDGE) {
    return;
  }

  auto const& conns = edge.static_lcons();
  auto it = std::lower_bound(std::begin(conns), std::end(conns), begin,
                             [](light_connection const& lcon, time const& t) {
                               return lcon.a_time_ < t;
                             });
  for (; it != std::end(conns) && it->a_time_ < end; ++it) {
    fun(it);
  }
}
*/

edge const* get_rt_route_edge(node const* route_node,
                              rt_light_connection const*, event_type);

node const* get_route_node(edge const&, event_type);

rt_light_connection const& get_rt_lcon(edge const* route_edge,
                                       size_t lcon_index);

time get_time(rt_light_connection const*, event_type);

time get_time(edge const* route_edge, std::size_t lcon_index, event_type);

lcon_idx_t get_lcon_index(edge const* route_edge, rt_light_connection const*);

}  // namespace motis
