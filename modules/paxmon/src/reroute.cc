#include "motis/paxmon/reroute.h"

#include <algorithm>
#include <iostream>
#include <set>

#include "utl/enumerate.h"
#include "utl/pairwise.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/graph_access.h"

namespace motis::paxmon {

trip_ev_key to_trip_ev_key(event_node* n) {
  return {n->station_, n->schedule_time_, n->type_, n};
}

std::vector<trip_ev_key> to_trip_ev_keys(trip_data const& td, graph const& g) {
  std::vector<trip_ev_key> teks;
  teks.reserve(td.edges_.size() * 2);
  for (auto const e : td.edges_) {
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
                          graph const& g) {
  if (from == nullptr || to == nullptr) {
    return nullptr;
  }
  for (auto const& e : from->outgoing_edges(g)) {
    if (e->to_ == to) {
      return e.get();
    }
  }
  return nullptr;
}

edge* connect_nodes(event_node* from, event_node* to, trip const* trp,
                    std::uint16_t capacity, graph const& g) {
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
  return add_edge(make_trip_edge(from, to, type, trp, capacity));
}

event_node* get_or_insert_node(graph& g, trip_data& td, trip_ev_key const tek,
                               std::set<event_node*>& reactivated_nodes) {
  for (auto const n : td.canceled_nodes_) {
    if (n->station_ == tek.station_id_ &&
        n->schedule_time_ == tek.schedule_time_ && n->type_ == tek.type_) {
      n->valid_ = true;
      reactivated_nodes.insert(n);
      return n;
    }
  }
  return g.nodes_
      .emplace_back(std::make_unique<event_node>(event_node{tek.schedule_time_,
                                                            tek.schedule_time_,
                                                            tek.type_,
                                                            true,
                                                            tek.station_id_,
                                                            {},
                                                            {}}))
      .get();
}

std::uint16_t guess_trip_capacity(schedule const& sched, paxmon_data& data,
                                  trip const* trp) {
  auto const sections = access::sections(trp);
  if (begin(sections) != end(sections)) {
    return get_capacity(sched, (*begin(sections)).lcon(),
                        data.trip_capacity_map_, data.category_capacity_map_,
                        data.default_capacity_);
  } else {
    return 0;
  }
}

std::set<passenger_group*> collect_passenger_groups(trip_data& td) {
  std::set<passenger_group*> affected_passenger_groups;
  for (auto const& te : td.edges_) {
    for (auto& psi : te->pax_connection_info_.section_infos_) {
      if (psi.valid_) {
        affected_passenger_groups.insert(psi.group_);
        psi.valid_ = false;
        te->passengers_ -= psi.group_->passengers_;
      }
    }
  }
  return affected_passenger_groups;
}

bool update_passenger_group(trip_data& td, extern_trip const& et,
                            passenger_group* pg, graph const& g) {
  static constexpr auto const INVALID_INDEX =
      std::numeric_limits<std::size_t>::max();
  for (auto const& leg : pg->compact_planned_journey_.legs_) {
    if (leg.trip_ == et) {
      auto enter_index = INVALID_INDEX;
      auto exit_index = INVALID_INDEX;
      for (auto const [idx, e] : utl::enumerate(td.edges_)) {
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
          add_passenger_group_to_edge(td.edges_[idx], pg);
        }
        return true;
      }
      return false;
    }
  }
  return false;
}

void apply_reroute(paxmon_data& data, schedule const& sched, trip const* trp,
                   extern_trip const& et, trip_data& td,
                   std::vector<trip_ev_key> const& old_route,
                   std::vector<trip_ev_key> const& new_route,
                   std::vector<edge*>& updated_interchange_edges) {
  auto const capacity = guess_trip_capacity(sched, data, trp);
  auto const affected_passenger_groups = collect_passenger_groups(td);
  auto diff = diff_route(old_route, new_route);

  std::vector<event_node*> new_nodes;
  std::set<event_node*> removed_nodes;
  std::set<event_node*> reactivated_nodes;

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
            get_or_insert_node(data.graph_, td, tek, reactivated_nodes);
        new_nodes.emplace_back(new_node);
        break;
      }
    }
  }

  std::vector<edge*> new_edges;
  for (auto const& [from, to] : utl::pairwise(new_nodes)) {
    auto e = connect_nodes(from, to, trp, capacity, data.graph_);
    new_edges.emplace_back(e);
  }
  td.edges_ = new_edges;
  std::copy(begin(removed_nodes), end(removed_nodes),
            std::back_inserter(td.canceled_nodes_));

  for (auto pg : affected_passenger_groups) {
    update_passenger_group(td, et, pg, data.graph_);
  }

  for (auto const n : removed_nodes) {
    for (auto const& e : n->outgoing_edges(data.graph_)) {
      if (e->type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(e.get());
      }
    }
    for (auto const& e : n->incoming_edges(data.graph_)) {
      if (e->type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(e);
      }
    }
  }

  for (auto const n : reactivated_nodes) {
    for (auto const& e : n->outgoing_edges(data.graph_)) {
      if (e->type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(e.get());
      }
    }
    for (auto const& e : n->incoming_edges(data.graph_)) {
      if (e->type_ == edge_type::INTERCHANGE) {
        updated_interchange_edges.emplace_back(e);
      }
    }
  }
}

}  // namespace motis::paxmon
