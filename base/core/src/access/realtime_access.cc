#include "motis/core/access/realtime_access.h"

#include <cassert>

#include "motis/core/access/edge_access.h"

namespace motis {

time get_schedule_time(schedule const& sched, ev_key const& k) {
  auto it = sched.graph_to_delay_info_.find(k);
  if (it == end(sched.graph_to_delay_info_)) {
    return get_time(k.route_edge_, k.lcon_idx_, k.ev_type_);
  } else {
    return it->second->get_schedule_time();
  }
}

time get_schedule_time(schedule const& sched, edge const* route_edge,
                       lcon_idx_t const lcon_index, event_type const ev_type) {
  auto it = sched.graph_to_delay_info_.find({route_edge, lcon_index, ev_type});
  if (it == end(sched.graph_to_delay_info_)) {
    return get_time(route_edge, lcon_index, ev_type);
  } else {
    return it->second->get_schedule_time();
  }
}

time get_schedule_time(schedule const& sched, edge const* route_edge,
                       light_connection const* lcon, event_type const ev_type) {
  auto it = sched.graph_to_delay_info_.find(
      {route_edge, get_lcon_index(route_edge, lcon), ev_type});
  if (it == end(sched.graph_to_delay_info_)) {
    return ev_type == event_type::DEP ? lcon->d_time_ : lcon->a_time_;
  } else {
    return it->second->get_schedule_time();
  }
}

time get_delay(schedule const& sched, ev_key const& k) {
  return get_time(k.lcon(), k.ev_type_) - get_schedule_time(sched, k);
}

delay_info get_delay_info(schedule const& sched, node const* route_node,
                          light_connection const* lcon,
                          event_type const ev_type) {
  auto route_edge = get_route_edge(route_node, lcon, ev_type);
  auto lcon_idx = get_lcon_index(route_edge, lcon);
  auto it = sched.graph_to_delay_info_.find({route_edge, lcon_idx, ev_type});
  if (it == end(sched.graph_to_delay_info_)) {
    return delay_info{ev_key{route_edge, lcon_idx, ev_type}};
  } else {
    return *it->second;
  }
}

delay_info get_delay_info(schedule const& sched, edge const* route_edge,
                          light_connection const* lcon,
                          event_type const ev_type) {
  auto lcon_idx = get_lcon_index(route_edge, lcon);
  auto it = sched.graph_to_delay_info_.find({route_edge, lcon_idx, ev_type});
  if (it == end(sched.graph_to_delay_info_)) {
    return delay_info{ev_key{route_edge, lcon_idx, ev_type}};
  } else {
    return *it->second;
  }
}

delay_info get_delay_info(schedule const& sched, ev_key const& k) {
  auto it = sched.graph_to_delay_info_.find(k);
  if (it == end(sched.graph_to_delay_info_)) {
    return delay_info{k};
  } else {
    return *it->second;
  }
}

ev_key const& get_current_ev_key(schedule const& sched, ev_key const& k) {
  auto const it = sched.graph_to_delay_info_.find(k);
  if (it != end(sched.graph_to_delay_info_)) {
    return it->second->get_ev_key();
  } else {
    return k;
  }
}

ev_key const& get_orig_ev_key(schedule const& sched, ev_key const& k) {
  auto const it = sched.graph_to_delay_info_.find(k);
  if (it != end(sched.graph_to_delay_info_)) {
    return it->second->get_orig_ev_key();
  } else {
    return k;
  }
}

uint16_t get_schedule_track(schedule const& sched, ev_key const& k) {
  auto it = sched.graph_to_schedule_track_index_.find(k);
  if (it == end(sched.graph_to_schedule_track_index_)) {
    auto const full_con = k.lcon()->full_con_;
    return k.is_arrival() ? full_con->a_track_ : full_con->d_track_;
  } else {
    return it->second;
  }
}

int get_schedule_track(schedule const& sched, edge const* route_edge,
                       light_connection const* lcon, event_type const ev_type) {
  return get_schedule_track(
      sched, {route_edge, get_lcon_index(route_edge, lcon), ev_type});
}

int get_schedule_track(schedule const& sched, node const* route_node,
                       light_connection const* lcon, event_type const ev_type) {
  auto route_edge = get_route_edge(route_node, lcon, ev_type);
  auto lcon_idx = get_lcon_index(route_edge, lcon);
  return get_schedule_track(sched, {route_edge, lcon_idx, ev_type});
}

}  // namespace motis
