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

namespace motis::paxmon {

trip_ev_key to_trip_ev_key(event_node* n) {
  return {n->station_, n->schedule_time_, n->type_, n};
}

std::vector<trip_ev_key> to_trip_ev_keys(trip_data_index tdi, graph& g) {
  std::vector<trip_ev_key> teks;
  auto const edges = g.trip_data_.edges(tdi);
  teks.reserve(edges.size() * 2);
  for (auto const& ei : edges) {
    auto const* e = ei.get(g);
    teks.emplace_back(to_trip_ev_key(e->from(g)));
    teks.emplace_back(to_trip_ev_key(e->to(g)));
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
                          graph& g) {
  if (from == nullptr || to == nullptr) {
    return nullptr;
  }
  for (auto& e : from->outgoing_edges(g)) {
    if (e.to_ == to->index_) {
      return &e;
    }
  }
  return nullptr;
}

edge* connect_nodes(event_node* from, event_node* to,
                    merged_trips_idx merged_trips,
                    std::uint16_t encoded_capacity, graph& g) {
  if (from == nullptr || to == nullptr) {
    return nullptr;
  }
  utl::verify(
      (from->type_ == event_type::DEP && to->type_ == event_type::ARR) ||
          (from->type_ == event_type::ARR && to->type_ == event_type::DEP),
      "invalid event sequence");
  if (auto e = get_connecting_edge(from, to, g); e != nullptr) {
    return e;
  }
  auto const type =
      from->type_ == event_type::DEP ? edge_type::TRIP : edge_type::WAIT;
  auto const cap = from->type_ == event_type::DEP ? encoded_capacity
                                                  : UNLIMITED_ENCODED_CAPACITY;
  return add_edge(
      g, make_trip_edge(g, from->index_, to->index_, type, merged_trips, cap,
                        service_class::OTHER));  // TODO(pablo): service class
}

event_node* get_or_insert_node(graph& g, trip_data_index const tdi,
                               trip_ev_key const tek,
                               std::set<event_node*>& reactivated_nodes) {
  for (auto const ni : g.trip_data_.canceled_nodes(tdi)) {
    auto const n = &g.graph_.nodes_[ni];
    if (n->station_ == tek.station_id_ &&
        n->schedule_time_ == tek.schedule_time_ && n->type_ == tek.type_) {
      n->valid_ = true;
      reactivated_nodes.insert(n);
      return n;
    }
  }
  return &g.graph_.emplace_back_node(
      static_cast<event_node_index>(g.graph_.nodes_.size()), tek.schedule_time_,
      tek.schedule_time_, tek.type_, true, tek.station_id_);
}

std::pair<std::uint16_t, capacity_source> guess_trip_capacity(
    schedule const& sched, capacity_maps const& caps, trip const* trp) {
  auto const sections = access::sections(trp);
  if (begin(sections) != end(sections)) {
    return get_capacity(sched, (*begin(sections)).lcon(),
                        caps.trip_capacity_map_, caps.category_capacity_map_);
  } else {
    return {UNKNOWN_CAPACITY, capacity_source::SPECIAL};
  }
}

std::set<passenger_group*> collect_passenger_groups(graph& g,
                                                    trip_data_index const tdi) {
  std::set<passenger_group*> affected_passenger_groups;
  for (auto const& tei : g.trip_data_.edges(tdi)) {
    auto* te = tei.get(g);
    auto groups = g.pax_connection_info_.groups_[te->pci_];
    for (auto pg_id : groups) {
      auto* pg = g.passenger_groups_[pg_id];
      affected_passenger_groups.insert(pg);
      utl::erase(pg->edges_, tei);
    }
    groups.clear();
  }
  return affected_passenger_groups;
}

bool update_passenger_group(trip_data_index const tdi, trip const* trp,
                            passenger_group* pg, graph& g) {
  static constexpr auto const INVALID_INDEX =
      std::numeric_limits<std::size_t>::max();
  for (auto const& leg : pg->compact_planned_journey_.legs_) {
    if (leg.trip_ == trp) {
      auto const edges = g.trip_data_.edges(tdi);
      auto enter_index = INVALID_INDEX;
      auto exit_index = INVALID_INDEX;
      for (auto const& [idx, ei] : utl::enumerate(edges)) {
        auto const e = ei.get(g);
        auto const from = e->from(g);
        auto const to = e->to(g);
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
          auto* e = ei.get(g);
          add_passenger_group_to_edge(g, e, pg);
          pg->edges_.emplace_back(ei);
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

void apply_reroute(paxmon_data& data, capacity_maps const& caps, schedule const& sched, trip const* trp,
                   trip_data_index const tdi,
                   std::vector<trip_ev_key> const& old_route,
                   std::vector<trip_ev_key> const& new_route,
                   std::vector<edge_index>& updated_interchange_edges) {
  auto const encoded_capacity =
      encode_capacity(guess_trip_capacity(sched, caps, trp));
  auto const affected_passenger_groups =
      collect_passenger_groups(data.graph_, tdi);
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
        break;
      }

      case diff_op::INSERT: {
        auto new_node =
            get_or_insert_node(data.graph_, tdi, tek, reactivated_nodes);
        new_nodes.emplace_back(new_node);
        break;
      }
    }
  }

  auto edges = data.graph_.trip_data_.edges(tdi);
  edges.clear();
  if (!new_nodes.empty()) {
    auto const merged_trips = get_merged_trips(trp).value();
    for (auto const& [from, to] : utl::pairwise(new_nodes)) {
      auto e =
          connect_nodes(from, to, merged_trips, encoded_capacity, data.graph_);
      if (e->is_trip()) {
        edges.emplace_back(get_edge_index(data.graph_, e));
      }
    }
  }
  auto canceled_nodes = data.graph_.trip_data_.canceled_nodes(tdi);
  for (auto const* n : removed_nodes) {
    canceled_nodes.emplace_back(n->index(data.graph_));
  }

  for (auto pg : affected_passenger_groups) {
    update_passenger_group(tdi, trp, pg, data.graph_);
  }

  for (auto* n : removed_nodes) {
    for (auto& e : n->outgoing_edges(data.graph_)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(data.graph_, &e));
      }
    }
    for (auto& e : n->incoming_edges(data.graph_)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(data.graph_, &e));
      }
    }
  }

  for (auto* n : reactivated_nodes) {
    for (auto& e : n->outgoing_edges(data.graph_)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(data.graph_, &e));
      }
    }
    for (auto& e : n->incoming_edges(data.graph_)) {
      if (e.type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(get_edge_index(data.graph_, &e));
      }
    }
  }
}

}  // namespace motis::paxmon
