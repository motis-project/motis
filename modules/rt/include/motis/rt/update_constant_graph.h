#pragma once

#include <cassert>

#include "motis/core/schedule/schedule.h"

namespace motis::rt {

inline void constant_graph_add_station_node(schedule& sched) {
  sched.travel_time_lower_bounds_fwd_.resize(sched.station_nodes_.size());
  sched.travel_time_lower_bounds_bwd_.resize(sched.station_nodes_.size());
}

inline void constant_graph_add_route_node(schedule& sched, int route_index,
                                          station_node const* sn,
                                          bool in_allowed, bool out_allowed) {
  auto const route_offset =
      static_cast<uint32_t>(sched.non_station_node_offset_);
  auto const route_lb_node_id =
      route_offset + static_cast<uint32_t>(route_index);
  auto const cg_size = route_offset + sched.route_count_;

  auto const add_edge = [&](uint32_t const from, uint32_t const to,
                            bool const is_exit) {
    auto& fwd_edges = sched.transfers_lower_bounds_fwd_[to];
    if (std::find_if(begin(fwd_edges), end(fwd_edges), [&](auto const& se) {
          return se.to_ == from;
        }) == end(fwd_edges)) {
      fwd_edges.emplace_back(from, is_exit);
    }

    auto& bwd_edges = sched.transfers_lower_bounds_bwd_[from];
    if (std::find_if(begin(bwd_edges), end(bwd_edges), [&](auto const& se) {
          return se.to_ == to;
        }) == end(bwd_edges)) {
      bwd_edges.emplace_back(to, !is_exit);
    }
  };

  sched.transfers_lower_bounds_fwd_.resize(cg_size);
  sched.transfers_lower_bounds_bwd_.resize(cg_size);

  if (in_allowed) {
    add_edge(sn->id_, route_lb_node_id, false);
  }
  if (out_allowed) {
    add_edge(route_lb_node_id, sn->id_, true);
  }
}

inline void constant_graph_add_route_edge(
    schedule& sched, trip_info::route_edge const& route_edge) {
  auto const min_cost = route_edge->get_minimum_cost();
  if (!min_cost.is_valid()) {
    return;
  }

  auto const update_min = [&](constant_graph& cg, uint32_t const from,
                              uint32_t const to) {
    assert(from < cg.size() && to < cg.size());
    for (auto& se : cg[from]) {
      if (se.to_ == to) {
        se.cost_ = std::min(se.cost_, min_cost.time_);
        return;
      }
    }
    cg[from].emplace_back(to, min_cost.time_);
  };

  auto const from_station_id = route_edge->from_->get_station()->id_;
  auto const to_station_id = route_edge->to_->get_station()->id_;

  update_min(sched.travel_time_lower_bounds_fwd_, to_station_id,
             from_station_id);
  update_min(sched.travel_time_lower_bounds_bwd_, from_station_id,
             to_station_id);
}

}  // namespace motis::rt
