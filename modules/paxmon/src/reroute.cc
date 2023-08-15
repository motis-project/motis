#include "motis/paxmon/reroute.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>

#include "utl/enumerate.h"
#include "utl/erase.h"
#include "utl/pairwise.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/edge_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/graph_access.h"
#include "motis/paxmon/update_capacity.h"

namespace motis::paxmon {

trip_ev_key to_trip_ev_key(event_node* n) {
  return {n->station_, n->schedule_time_, n->type_, n};
}

std::vector<trip_ev_key> to_trip_ev_keys(trip_data_index tdi, universe& uv) {
  std::vector<trip_ev_key> teks;
  auto const edges = uv.trip_data_.edges(tdi);
  teks.reserve(edges.size() * 2);
  for (auto const& ei : edges) {
    auto const* e = ei.get(uv);
    teks.emplace_back(to_trip_ev_key(e->from(uv)));
    teks.emplace_back(to_trip_ev_key(e->to(uv)));
  }
  return teks;
}

std::vector<trip_ev_key> to_trip_ev_keys(
    schedule const& sched,
    flatbuffers::Vector<flatbuffers::Offset<motis::rt::RtEventInfo>> const&
        events) {
  return utl::to_vec(events, [&](motis::rt::RtEventInfo const* ei) {
    return trip_ev_key{
        get_station_node(sched, ei->station_id()->str())->id_,
        unix_to_motistime(sched.schedule_begin_, ei->schedule_time()),
        ei->event_type() == EventType_DEP ? event_type::DEP : event_type::ARR,
        nullptr};
  });
}

std::vector<std::pair<diff_op, trip_ev_key>> diff_route(
    std::vector<trip_ev_key> const& old_route,
    std::vector<trip_ev_key> const& new_route) {
  std::vector<std::pair<diff_op, trip_ev_key>> diff;
  auto old_start = 0ULL;
  auto new_start = 0ULL;
start:
  for (auto old_idx = old_start; old_idx < old_route.size(); ++old_idx) {
    auto const& old_tek = old_route[old_idx];
    for (auto new_idx = new_start; new_idx < new_route.size(); ++new_idx) {
      auto const& new_tek = new_route[new_idx];
      if (old_tek == new_tek) {
        for (auto i = old_start; i < old_idx; ++i) {
          diff.emplace_back(diff_op::REMOVE, old_route[i]);
        }
        for (auto i = new_start; i < new_idx; ++i) {
          diff.emplace_back(diff_op::INSERT, new_route[i]);
        }
        diff.emplace_back(diff_op::KEEP, old_route[old_idx]);
        old_start = old_idx + 1;
        new_start = new_idx + 1;
        goto start;
      }
    }
  }
  for (auto i = old_start; i < old_route.size(); ++i) {
    diff.emplace_back(diff_op::REMOVE, old_route[i]);
  }
  for (auto i = new_start; i < new_route.size(); ++i) {
    diff.emplace_back(diff_op::INSERT, new_route[i]);
  }

  return diff;
}

edge* get_connecting_edge(event_node const* from, event_node const* to,
                          universe& uv) {
  if (from == nullptr || to == nullptr) {
    return nullptr;
  }
  for (auto& e : from->outgoing_edges(uv)) {
    if (e.to_ == to->index_) {
      return &e;
    }
  }
  return nullptr;
}

inline void log_edge_update(universe& uv, schedule const& sched,
                            edge const& e) {
  if (uv.graph_log_.enabled_) {
    uv.graph_log_.edge_log_[e.pci_].emplace_back(edge_log_entry{
        sched.system_time_, e.transfer_time(),
        edge_log_entry::INVALID_TRANSFER_TIME, e.type_, e.broken_});
  }
}

// the following functions are split because otherwise clang-tidy complains
// that begin(from->outgoing_edges(uv)) allegedly returns nullptr

void disable_outgoing_edges(universe& uv, schedule const& sched,
                            event_node* from, edge const* except) {
  for (auto& e : from->outgoing_edges(uv)) {
    if (&e != except && (e.is_trip() || e.is_wait())) {
      e.type_ = edge_type::DISABLED;
      log_edge_update(uv, sched, e);
    }
  }
}

void disable_outgoing_edges(universe& uv, schedule const& sched,
                            event_node* from) {
  for (auto& e : from->outgoing_edges(uv)) {
    if (e.is_trip() || e.is_wait()) {
      e.type_ = edge_type::DISABLED;
      log_edge_update(uv, sched, e);
    }
  }
}

void disable_incoming_edges(universe& uv, schedule const& sched, event_node* to,
                            edge const* except) {
  for (auto& e : to->incoming_edges(uv)) {
    if (&e != except && (e.is_trip() || e.is_wait())) {
      e.type_ = edge_type::DISABLED;
      log_edge_update(uv, sched, e);
    }
  }
}

void disable_incoming_edges(universe& uv, schedule const& sched,
                            event_node* to) {
  for (auto& e : to->incoming_edges(uv)) {
    if (e.is_trip() || e.is_wait()) {
      e.type_ = edge_type::DISABLED;
      log_edge_update(uv, sched, e);
    }
  }
}

edge* connect_nodes(event_node* from, event_node* to,
                    merged_trips_idx merged_trips, std::uint16_t capacity,
                    capacity_source cap_source, universe& uv,
                    schedule const& sched) {
  if (from == nullptr || to == nullptr) {
    return nullptr;
  }
  utl::verify(
      (from->type_ == event_type::DEP && to->type_ == event_type::ARR) ||
          (from->type_ == event_type::ARR && to->type_ == event_type::DEP),
      "invalid event sequence");
  auto const type =
      from->type_ == event_type::DEP ? edge_type::TRIP : edge_type::WAIT;
  if (auto e = get_connecting_edge(from, to, uv); e != nullptr) {
    if (e->is_disabled()) {
      e->type_ = type;
      log_edge_update(uv, sched, *e);
    }
    disable_outgoing_edges(uv, sched, from, e);
    disable_incoming_edges(uv, sched, to, e);
    return e;
  }
  disable_outgoing_edges(uv, sched, from);
  disable_incoming_edges(uv, sched, to);
  auto const edge_cap =
      from->type_ == event_type::DEP ? capacity : UNLIMITED_CAPACITY;
  auto const edge_cap_source =
      from->type_ == event_type::DEP ? cap_source : capacity_source::UNLIMITED;
  return add_edge(
      uv, make_trip_edge(uv, from->index_, to->index_, type, merged_trips,
                         edge_cap, edge_cap_source,
                         service_class::OTHER));  // TODO(pablo): service class
}

event_node* get_or_insert_node(universe& uv, schedule const& sched,
                               trip_data_index const tdi, trip_ev_key const tek,
                               std::set<event_node*>& reactivated_nodes) {
  for (auto const ni : uv.trip_data_.canceled_nodes(tdi)) {
    auto const n = &uv.graph_.nodes_[ni];
    if (n->station_ == tek.station_id_ &&
        n->schedule_time_ == tek.schedule_time_ && n->type_ == tek.type_) {
      n->valid_ = true;
      reactivated_nodes.insert(n);
      if (uv.graph_log_.enabled_) {
        uv.graph_log_.node_log_[n->index_].emplace_back(
            node_log_entry{sched.system_time_, n->time_, n->valid_});
      }
      return n;
    }
  }
  return &uv.graph_.emplace_back_node(
      static_cast<event_node_index>(uv.graph_.nodes_.size()),
      tek.schedule_time_, tek.schedule_time_, tek.type_, true, tek.station_id_);
}

std::set<passenger_group_with_route> collect_and_remove_group_routes(
    universe& uv, schedule const& sched, trip_data_index const tdi) {
  std::set<passenger_group_with_route> affected_group_routes;
  for (auto const& tei : uv.trip_data_.edges(tdi)) {
    auto* te = tei.get(uv);
    auto group_routes = uv.pax_connection_info_.group_routes(te->pci_);
    for (auto const& pgwr : group_routes) {
      auto& route = uv.passenger_groups_.route(pgwr);
      auto edges = uv.passenger_groups_.route_edges(route.edges_index_);
      utl::verify(edges.empty() == route.disabled_,
                  "reroute[collect_and_remove_group_routes]: initial mismatch: "
                  "route_edges={}, disabled={}",
                  edges.size(), route.disabled_);
      affected_group_routes.insert(pgwr);
      utl::erase(edges, tei);
      if (edges.empty()) {
        route.disabled_ = true;
      }
    }
    if (uv.graph_log_.enabled_) {
      auto log = uv.graph_log_.pci_log_[te->pci_];
      for (auto const& pgwr : group_routes) {
        log.emplace_back(pci_log_entry{sched.system_time_,
                                       pci_log_action_t::ROUTE_REMOVED,
                                       pci_log_reason_t::TRIP_REROUTE, pgwr});
      }
    }
    group_routes.clear();
  }
  return affected_group_routes;
}

bool update_group_route(trip_data_index const tdi, trip const* trp,
                        passenger_group_with_route const pgwr, universe& uv,
                        schedule const& sched) {
  static constexpr auto const INVALID_INDEX =
      std::numeric_limits<std::size_t>::max();
  // TODO(pablo): does not support merged trips
  auto& gr = uv.passenger_groups_.route(pgwr);
  auto const cj = uv.passenger_groups_.journey(gr.compact_journey_index_);
  auto route_edges = uv.passenger_groups_.route_edges(gr.edges_index_);

  utl::verify(route_edges.empty() == gr.disabled_,
              "reroute[update_group_route]: initial mismatch: route_edges={}, "
              "disabled={}",
              route_edges.size(), gr.disabled_);

  for (auto const& leg : cj.legs()) {
    if (leg.trip_idx_ == trp->trip_idx_) {
      auto const edges = uv.trip_data_.edges(tdi);
      auto enter_index = INVALID_INDEX;
      auto exit_index = INVALID_INDEX;
      for (auto const& [idx, ei] : utl::enumerate(edges)) {
        auto const e = ei.get(uv);
        auto const from = e->from(uv);
        auto const to = e->to(uv);
        if (from->station_ == leg.enter_station_id_ &&
            from->schedule_time_ == leg.enter_time_) {
          enter_index = idx;
        } else if (to->station_ == leg.exit_station_id_ &&
                   to->schedule_time_ == leg.exit_time_) {
          exit_index = idx;
          break;
        }
      }
      if (enter_index != INVALID_INDEX && exit_index != INVALID_INDEX) {
        for (auto idx = enter_index; idx <= exit_index; ++idx) {
          auto const& ei = edges[idx];
          auto* e = ei.get(uv);
          add_group_route_to_edge(uv, sched, e, pgwr, true,
                                  pci_log_reason_t::TRIP_REROUTE);
          route_edges.emplace_back(ei);
        }
        if (!route_edges.empty()) {
          gr.disabled_ = false;
        }
        return true;
      }
      return false;
    }
  }
  return false;
}

std::optional<merged_trips_idx> get_merged_trips(trip const* trp) {
  if (trp->edges_->empty()) {
    return {};
  }
  return get_lcon(trp->edges_->front().get_edge(), trp->lcon_idx_).trips_;
}

void apply_reroute(universe& uv, schedule const& sched, trip const* trp,
                   trip_data_index const tdi,
                   std::vector<trip_ev_key> const& old_route,
                   std::vector<trip_ev_key> const& new_route,
                   std::vector<edge_index>& updated_interchange_edges) {
  auto const affected_group_routes =
      collect_and_remove_group_routes(uv, sched, tdi);
  auto diff = diff_route(old_route, new_route);

  std::vector<event_node*> new_nodes;
  std::set<event_node*> removed_nodes;
  std::set<event_node*>
      reactivated_nodes;  // TODO(pablo): remove from td.canceled_nodes?

  for (auto const& [op, tek] : diff) {
    switch (op) {
      case diff_op::KEEP: {
        new_nodes.emplace_back(tek.node_);
        break;
      }

      case diff_op::REMOVE: {
        tek.node_->valid_ = false;
        removed_nodes.insert(tek.node_);
        if (uv.graph_log_.enabled_) {
          uv.graph_log_.node_log_[tek.node_->index_].emplace_back(
              node_log_entry{sched.system_time_, tek.node_->time_,
                             tek.node_->valid_});
        }
        break;
      }

      case diff_op::INSERT: {
        auto new_node =
            get_or_insert_node(uv, sched, tdi, tek, reactivated_nodes);
        new_nodes.emplace_back(new_node);
        break;
      }
    }
  }

  auto edges = uv.trip_data_.edges(tdi);
  edges.clear();
  if (!new_nodes.empty()) {
    auto const merged_trips = get_merged_trips(trp).value();
    for (auto const& [from, to] : utl::pairwise(new_nodes)) {
      auto e = connect_nodes(from, to, merged_trips, UNKNOWN_CAPACITY,
                             capacity_source::UNKNOWN, uv, sched);
      if (e->is_trip()) {
        edges.emplace_back(get_edge_index(uv, e));
      }
    }
  }
  auto canceled_nodes = uv.trip_data_.canceled_nodes(tdi);
  for (auto const* n : removed_nodes) {
    canceled_nodes.emplace_back(n->index(uv));
  }

  update_trip_capacity(uv, sched, trp);

  for (auto const& pgwr : affected_group_routes) {
    update_group_route(tdi, trp, pgwr, uv, sched);
  }

  for (auto* n : removed_nodes) {
    for (auto& e : n->outgoing_edges(uv)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(uv, &e));
      }
    }
    for (auto& e : n->incoming_edges(uv)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(uv, &e));
      }
    }
  }

  for (auto* n : reactivated_nodes) {
    for (auto& e : n->outgoing_edges(uv)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(uv, &e));
      }
    }
    for (auto& e : n->incoming_edges(uv)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(uv, &e));
      }
    }
  }
}

}  // namespace motis::paxmon
