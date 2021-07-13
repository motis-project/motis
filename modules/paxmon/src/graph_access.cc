#include "motis/paxmon/graph_access.h"

#include <cassert>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>

#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pipes.h"
#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/reroute.h"

namespace motis::paxmon {

using namespace motis::rt;

struct rule_trip_adder {
  using rule_node_key =
      std::tuple<uint32_t /* station_idx */, time /* schedule_time */,
                 uint32_t /* merged_trips_idx */>;

  rule_trip_adder(schedule const& sched, paxmon_data& data)
      : sched_{sched}, data_{data} {}

  trip_data_index add_trip(trip const* trp) {
    if (auto const [_, inserted] = trips_.insert(trp); !inserted) {
      return INVALID_TRIP_DATA_INDEX;
    }

    utl::verify(!data_.graph_.trip_data_.contains(trp),
                "trip data already exists");

    auto const enter_exit_node_idx =
        static_cast<event_node_index>(data_.graph_.graph_.nodes_.size());
    data_.graph_.graph_.emplace_back_node(enter_exit_node_idx);

    auto const tdi =
        data_.graph_.trip_data_.insert_trip(trp, enter_exit_node_idx);
    auto trip_edges = data_.graph_.trip_data_.edges(tdi);

    auto prev_node = INVALID_EVENT_NODE_INDEX;
    for (auto const& section : motis::access::sections(trp)) {
      auto dep_node = get_or_create_dep_node(section);
      auto arr_node = get_or_create_arr_node(section);
      auto const ei = get_or_create_trip_edge(section, dep_node, arr_node);
      trip_edges.emplace_back(ei);
      if (prev_node != INVALID_EVENT_NODE_INDEX) {
        get_or_create_wait_edge(section, prev_node, dep_node);
      }
      prev_node = arr_node;

      add_merged_services(section);
      add_through_services(section, dep_node, arr_node);
    }

    return tdi;
  }

  trip_data_index get_or_add_trip(trip const* trp) {
    if (auto td = add_trip(trp); td != INVALID_TRIP_DATA_INDEX) {
      return td;
    } else {
      return data_.graph_.trip_data_.get_index(trp);
    }
  }

  void add_merged_services(motis::access::trip_section const& section) {
    for (auto const& merged_trp :
         *sched_.merged_trips_.at(section.lcon().trips_)) {
      if (merged_trp != section.trip_) {
        add_trip(merged_trp);
      }
    }
  }

  void add_through_services(motis::access::trip_section const& section,
                            event_node_index dep_node,
                            event_node_index arr_node) {
    (void)section;
    (void)dep_node;
    (void)arr_node;
    (void)sched_;
    // TODO(pablo): NYI
  }

  event_node_index get_or_create_dep_node(
      motis::access::trip_section const& section) {
    auto const station_idx = section.from_station_id();
    auto const schedule_time = get_schedule_time(
        sched_, section.edge(), section.trip_->lcon_idx_, event_type::DEP);
    auto const merged_trips_idx = section.lcon().trips_;
    return utl::get_or_create(
        dep_nodes_, rule_node_key{station_idx, schedule_time, merged_trips_idx},
        [&]() {
          auto const idx =
              static_cast<event_node_index>(data_.graph_.graph_.nodes_.size());
          data_.graph_.graph_.emplace_back_node(idx, section.lcon().d_time_,
                                                schedule_time, event_type::DEP,
                                                true, station_idx);
          return idx;
        });
  }

  event_node_index get_or_create_arr_node(
      motis::access::trip_section const& section) {
    auto const station_idx = section.to_station_id();
    auto const schedule_time = get_schedule_time(
        sched_, section.edge(), section.trip_->lcon_idx_, event_type::ARR);
    auto const merged_trips_idx = section.lcon().trips_;
    return utl::get_or_create(
        arr_nodes_, rule_node_key{station_idx, schedule_time, merged_trips_idx},
        [&]() {
          auto const idx =
              static_cast<event_node_index>(data_.graph_.graph_.nodes_.size());
          data_.graph_.graph_.emplace_back_node(idx, section.lcon().a_time_,
                                                schedule_time, event_type::ARR,
                                                true, station_idx);
          return idx;
        });
  }

  edge_index get_or_create_trip_edge(motis::access::trip_section const& section,
                                     event_node_index dep_node,
                                     event_node_index arr_node) {
    return utl::get_or_create(trip_edges_, &section.lcon(), [&]() {
      auto const encoded_capacity = encode_capacity(
          get_capacity(sched_, section.lcon(), data_.trip_capacity_map_,
                       data_.category_capacity_map_));
      auto const* e = add_edge(
          data_.graph_, make_trip_edge(dep_node, arr_node, edge_type::TRIP,
                                       section.lcon().trips_, encoded_capacity,
                                       section.fcon().clasz_));
      return get_edge_index(data_.graph_, e);
    });
  }

  edge_index get_or_create_wait_edge(motis::access::trip_section const& section,
                                     event_node_index prev_node,
                                     event_node_index dep_node) {
    return utl::get_or_create(
        wait_edges_, std::make_pair(prev_node, dep_node), [&]() {
          auto const* e = add_edge(
              data_.graph_,
              make_trip_edge(prev_node, dep_node, edge_type::WAIT,
                             section.lcon().trips_, UNLIMITED_ENCODED_CAPACITY,
                             section.fcon().clasz_));
          return get_edge_index(data_.graph_, e);
        });
  }

  schedule const& sched_;
  paxmon_data& data_;
  std::set<trip const*> trips_;
  std::map<rule_node_key, event_node_index> dep_nodes_;
  std::map<rule_node_key, event_node_index> arr_nodes_;
  std::map<motis::light_connection const*, edge_index> trip_edges_;
  std::map<std::pair<event_node_index, event_node_index>, edge_index>
      wait_edges_;
};

trip_data_index add_trip(schedule const& sched, paxmon_data& data,
                         trip const* trp) {
  auto adder = rule_trip_adder{sched, data};
  return adder.add_trip(trp);
}

trip_data_index get_or_add_trip(schedule const& sched, paxmon_data& data,
                                trip const* trp) {
  if (auto const idx = data.graph_.trip_data_.find_index(trp);
      idx != INVALID_TRIP_DATA_INDEX) {
    return idx;
  } else {
    return add_trip(sched, data, trp);
  }
}

trip_data_index get_or_add_trip(schedule const& sched, paxmon_data& data,
                                extern_trip const& et) {
  return get_or_add_trip(sched, data, get_trip(sched, et));
}

void add_interchange_edges(event_node* evn,
                           std::vector<edge_index>& updated_interchange_edges,
                           graph& g, system_statistics& system_stats) {
  if (evn->type_ == event_type::ARR) {
    auto oe = evn->outgoing_edges(g);
    return utl::all(oe)  //
           | utl::remove_if(
                 [](auto&& e) { return e.type_ != edge_type::INTERCHANGE; })  //
           | utl::for_each([&](auto&& e) {
               ++system_stats.total_updated_interchange_edges_;
               updated_interchange_edges.push_back(get_edge_index(g, &e));
             });
  } else /*if (evn->type_ == event_type::DEP)*/ {
    assert(evn->type_ == event_type::DEP);
    return utl::all(evn->incoming_edges(g))  //
           | utl::remove_if(
                 [](auto&& e) { return e.type_ != edge_type::INTERCHANGE; })  //
           | utl::for_each([&](auto&& e) {
               ++system_stats.total_updated_interchange_edges_;
               updated_interchange_edges.push_back(get_edge_index(g, &e));
             });
  }
}

void update_event_times(schedule const& sched, graph& g,
                        RtDelayUpdate const* du,
                        std::vector<edge_index>& updated_interchange_edges,
                        system_statistics& system_stats) {
  auto const trp = from_fbs(sched, du->trip());
  auto const tdi = g.trip_data_.find_index(trp);
  if (tdi == INVALID_TRIP_DATA_INDEX) {
    return;
  }
  auto trip_edges = g.trip_data_.edges(tdi);
  ++system_stats.update_event_times_trip_edges_found_;
  for (auto const& ue : *du->events()) {
    auto const station_id =
        get_station(sched, ue->base()->station_id()->str())->index_;
    auto const schedule_time =
        unix_to_motistime(sched, ue->base()->schedule_time());
    for (auto tei : trip_edges) {
      auto const* te = tei.get(g);
      auto const from = te->from(g);
      auto const to = te->to(g);
      if (ue->base()->event_type() == EventType_DEP &&
          from->type_ == event_type::DEP && from->station_ == station_id &&
          from->schedule_time_ == schedule_time) {
        ++system_stats.update_event_times_dep_updated_;
        from->time_ =
            unix_to_motistime(sched.schedule_begin_, ue->updated_time());
        add_interchange_edges(from, updated_interchange_edges, g, system_stats);
      } else if (ue->base()->event_type() == EventType_ARR &&
                 to->type_ == event_type::ARR && to->station_ == station_id &&
                 to->schedule_time_ == schedule_time) {
        ++system_stats.update_event_times_arr_updated_;
        to->time_ =
            unix_to_motistime(sched.schedule_begin_, ue->updated_time());
        add_interchange_edges(to, updated_interchange_edges, g, system_stats);
      }
    }
  }
}

void update_trip_route(schedule const& sched, paxmon_data& data,
                       RtRerouteUpdate const* ru,
                       std::vector<edge_index>& updated_interchange_edges,
                       system_statistics& system_stats) {
  ++system_stats.update_trip_route_count_;
  auto const trp = from_fbs(sched, ru->trip());
  auto const tdi = data.graph_.trip_data_.find_index(trp);
  if (tdi == INVALID_TRIP_DATA_INDEX) {
    return;
  }
  auto trip_edges = data.graph_.trip_data_.edges(tdi);
  ++system_stats.update_trip_route_trip_edges_found_;

  auto const current_teks = to_trip_ev_keys(tdi, data.graph_);
  auto const new_teks = to_trip_ev_keys(sched, *ru->new_route());

  apply_reroute(data, sched, trp, tdi, current_teks, new_teks,
                updated_interchange_edges);
}

void add_passenger_group_to_edge(edge* e, passenger_group* pg) {
  auto& pci = e->get_pax_connection_info();
  pci.groups_.emplace(pg->id_);
  if (is_planned_group(pg)) {
    pci.expected_load_ += pg->passengers_;
  }
}

void remove_passenger_group_from_edge(edge* e, passenger_group* pg) {
  e->get_pax_connection_info().groups_.erase(pg->id_);
}

void for_each_trip(
    schedule const& sched, paxmon_data& data, compact_journey const& journey,
    std::function<void(journey_leg const&, trip_data_index)> const& fn) {
  for (auto const& leg : journey.legs_) {
    fn(leg, get_or_add_trip(sched, data, leg.trip_));
  }
}

void for_each_edge(schedule const& sched, paxmon_data& data,
                   compact_journey const& journey,
                   std::function<void(journey_leg const&, edge*)> const& fn) {
  for_each_trip(sched, data, journey,
                [&](journey_leg const& leg, trip_data_index const tdi) {
                  auto in_trip = false;
                  for (auto const& ei : data.graph_.trip_data_.edges(tdi)) {
                    auto* e = ei.get(data.graph_);
                    if (!in_trip) {
                      auto const from = e->from(data.graph_);
                      if (from->station_idx() == leg.enter_station_id_ &&
                          from->schedule_time() == leg.enter_time_) {
                        in_trip = true;
                      }
                    }
                    if (in_trip) {
                      fn(leg, e);
                      auto const to = e->to(data.graph_);
                      if (to->station_idx() == leg.exit_station_id_ &&
                          to->schedule_time() == leg.exit_time_) {
                        break;
                      }
                    }
                  }
                });
}

event_node* find_event_node(graph& g, trip_data_index const tdi,
                            std::uint32_t const station_idx,
                            event_type const et, time const schedule_time) {
  for (auto& ei : g.trip_data_.edges(tdi)) {
    auto* e = ei.get(g);
    if (et == event_type::DEP) {
      auto* n = e->from(g);
      if (n->station_idx() == station_idx &&
          n->schedule_time() == schedule_time) {
        return n;
      }
    } else if (et == event_type::ARR) {
      auto* n = e->to(g);
      if (n->station_idx() == station_idx &&
          n->schedule_time() == schedule_time) {
        return n;
      }
    }
  }
  return nullptr;
}

}  // namespace motis::paxmon
