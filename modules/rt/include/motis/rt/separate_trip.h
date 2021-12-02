#pragma once

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/core/schedule/build_platform_node.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/access/bfs.h"
#include "motis/core/access/service_access.h"

#include "motis/rt/in_out_allowed.h"
#include "motis/rt/incoming_edges.h"
#include "motis/rt/update_constant_graph.h"

namespace motis::rt {

inline std::set<trip::route_edge> route_edges(ev_key const& k) {
  return route_bfs(k, bfs_direction::BOTH, true);
}

inline std::map<node const*, in_out_allowed> get_route_in_out_allowed(
    ev_key const& k) {
  std::map<node const*, in_out_allowed> in_out_allowed;
  for (auto const& e : route_edges(k)) {
    in_out_allowed[e->from_] = get_in_out_allowed(e->from_);
    in_out_allowed[e->to_] = get_in_out_allowed(e->to_);
  }
  return in_out_allowed;
}

inline edge copy_edge(edge const& original, node* from, node* to,
                      int lcon_index) {
  edge e;
  if (original.type() == edge::ROUTE_EDGE) {
    e = make_route_edge(from, to, {original.m_.route_edge_.conns_[lcon_index]});
  } else {
    e = original;
    e.from_ = from;
    e.to_ = to;
  }
  return e;
}

inline void copy_trip_route(
    schedule& sched, ev_key const& k, std::map<node const*, node*>& nodes,
    std::map<trip::route_edge, trip::route_edge>& edges,
    std::map<node const*,
             std::pair<light_connection const*, light_connection const*>>&
        lcons) {
  auto const route_id = sched.route_count_++;

  auto const build_node = [&](node* orig) {
    return orig->station_node_->child_nodes_
        .emplace_back(mcd::make_unique<node>(
            make_node(node_type::ROUTE_NODE, orig->station_node_,
                      sched.next_node_id_++, route_id)))
        .get();
  };

  for (auto const& e : route_edges(k)) {
    auto const from = utl::get_or_create(nodes, e->from_,
                                         [&] { return build_node(e->from_); });
    auto const to =
        utl::get_or_create(nodes, e->to_, [&] { return build_node(e->to_); });

    from->edges_.push_back(copy_edge(*e, from, to, k.lcon_idx_));
    auto const& new_edge = from->edges_.back();
    edges[e] = trip::route_edge(&new_edge);
    constant_graph_add_route_edge(sched, edges[e]);

    if (e->type() == edge::ROUTE_EDGE) {
      auto const& old_lcon = e->m_.route_edge_.conns_[k.lcon_idx_];
      const_cast<light_connection&>(old_lcon).valid_ = false;  // NOLINT
      auto const& new_lcon = new_edge.m_.route_edge_.conns_.back();
      lcons[from].second = &new_lcon;
      lcons[to].first = &new_lcon;
    }
  }
}

inline std::set<trip const*> route_trips(schedule const& sched,
                                         ev_key const& k) {
  auto trips = std::set<trip const*>{};
  for (auto const& e : route_edges(k)) {
    if (e->empty()) {
      continue;
    }

    auto trips_idx = e->m_.route_edge_.conns_[k.lcon_idx_].trips_;
    auto const& merged_trips = *sched.merged_trips_.at(trips_idx);
    trips.insert(begin(merged_trips), end(merged_trips));
  }
  return trips;
}

inline void update_trips(schedule& sched, ev_key const& k,
                         std::map<trip::route_edge, trip::route_edge>& edges) {
  for (auto const& t : route_trips(sched, k)) {
    sched.trip_edges_.emplace_back(
        mcd::make_unique<mcd::vector<trip::route_edge>>(
            mcd::to_vec(*t->edges_, [&](trip::route_edge const& e) {
              return edges.at(e.get_edge());
            })));
    const_cast<trip*>(t)->edges_ = sched.trip_edges_.back().get();  // NOLINT
    const_cast<trip*>(t)->lcon_idx_ = 0;  // NOLINT
  }
}

inline void build_change_edges(
    schedule& sched,
    std::map<node const*, in_out_allowed> const& in_out_allowed,
    std::map<node const*, node*> const& nodes,
    std::map<node const*,
             std::pair<light_connection const*, light_connection const*>> const&
        lcons,
    std::vector<incoming_edge_patch>& incoming) {
  for (auto& n : nodes) {
    auto station_node =
        sched.station_nodes_.at(n.first->get_station()->id_).get();
    auto in_out = in_out_allowed.at(n.first);
    auto route_node = n.second;
    auto const station = sched.stations_.at(station_node->id_).get();
    auto const transfer_time = station->transfer_time_;

    if (!in_out.in_allowed_) {
      station_node->edges_.push_back(
          make_invalid_edge(station_node, route_node));
    } else {
      station_node->edges_.push_back(
          make_enter_edge(station_node, route_node, transfer_time, true));
      auto const lcon = lcons.at(route_node).second;
      if (lcon != nullptr) {
        auto const platform = station->get_platform(lcon->full_con_->d_track_);
        if (platform) {
          auto const pn = add_platform_enter_edge(
              sched, route_node, station_node, station->platform_transfer_time_,
              platform.value());
          add_outgoing_edge(&pn->edges_.back(), incoming);
        }
      }
    }
    add_outgoing_edge(&station_node->edges_.back(), incoming);

    if (!in_out.out_allowed_) {
      route_node->edges_.push_back(make_invalid_edge(route_node, station_node));
    } else {
      route_node->edges_.push_back(
          make_exit_edge(route_node, station_node, transfer_time, true));
      auto const lcon = lcons.at(route_node).first;
      if (lcon != nullptr) {
        auto const platform = station->get_platform(lcon->full_con_->a_track_);
        if (platform) {
          add_platform_exit_edge(sched, route_node, station_node,
                                 station->platform_transfer_time_,
                                 platform.value());
        }
      }
    }

    if (in_out.out_allowed_ && station_node->foot_node_) {
      route_node->edges_.push_back(make_after_train_fwd_edge(
          route_node, station_node->foot_node_.get(), 0, true));
      station_node->foot_node_->edges_.push_back(make_after_train_bwd_edge(
          station_node->foot_node_.get(), route_node, 0, true));
      add_outgoing_edge(&station_node->foot_node_->edges_.back(), incoming);
    }

    constant_graph_add_route_node(sched, route_node->route_, station_node,
                                  in_out.in_allowed_, in_out.out_allowed_);
  }
}

inline std::set<station_node*> route_station_nodes(ev_key const& k) {
  std::set<station_node*> station_nodes;
  for (auto const& e : route_edges(k)) {
    station_nodes.insert(e->from_->get_station());
    station_nodes.insert(e->to_->get_station());
  }
  return station_nodes;
}

inline void update_delays(
    lcon_idx_t const lcon_idx,
    std::map<trip::route_edge, trip::route_edge> const& edges,
    schedule& sched) {
  auto const update_di = [&](ev_key const& orig_k, ev_key const& new_k) {
    auto const it = sched.graph_to_delay_info_.find(orig_k);
    if (it != end(sched.graph_to_delay_info_)) {
      auto const di = it->second;
      sched.graph_to_delay_info_[new_k] = di;
      di->set_ev_key(new_k);
    }
  };

  for (auto const& entry : edges) {
    auto const e = entry.first;
    auto const new_e = entry.second.get_edge();

    auto const orig_dep = ev_key{e, lcon_idx, event_type::DEP};
    auto const new_dep = ev_key{new_e, 0, event_type::DEP};
    auto const orig_arr = ev_key{e, lcon_idx, event_type::ARR};
    auto const new_arr = ev_key{new_e, 0, event_type::ARR};

    update_di(orig_dep, new_dep);
    update_di(orig_arr, new_arr);
  }
}

inline void seperate_trip(schedule& sched, ev_key const& k) {
  auto const in_out_allowed = get_route_in_out_allowed(k);
  auto const station_nodes = route_station_nodes(k);
  std::vector<incoming_edge_patch> incoming;
  save_outgoing_edges(station_nodes, incoming);
  auto nodes = std::map<node const*, node*>{};
  auto edges = std::map<trip::route_edge, trip::route_edge>{};
  auto lcons =
      std::map<node const*,
               std::pair<light_connection const*, light_connection const*>>{};

  copy_trip_route(sched, k, nodes, edges, lcons);
  update_trips(sched, k, edges);
  build_change_edges(sched, in_out_allowed, nodes, lcons, incoming);
  add_outgoing_edges_from_new_route(nodes, incoming);
  patch_incoming_edges(incoming);
  update_delays(k.lcon_idx_, edges, sched);
}

inline void seperate_trip(schedule& sched, trip const* trp) {
  auto const first_dep =
      ev_key{trp->edges_->front().get_edge(), trp->lcon_idx_, event_type::DEP};
  seperate_trip(sched, first_dep);
}

}  // namespace motis::rt
