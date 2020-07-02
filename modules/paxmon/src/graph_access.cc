#include "motis/paxmon/graph_access.h"

#include <cassert>
#include <algorithm>
#include <iostream>
#include <set>
#include <string_view>

#include "utl/get_or_create.h"
#include "utl/pipes.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/reroute.h"

namespace motis::paxmon {

using namespace motis::rt;

std::vector<edge*> add_trip(schedule const& sched, paxmon_data& data,
                            extern_trip const& et) {
  std::vector<edge*> edges;

  auto trp = get_trip(sched, et);
  event_node* prev_node = nullptr;
  for (auto const& section : motis::access::sections(trp)) {
    auto const& lc = section.lcon();
    auto dep_node = data.graph_.nodes_
                        .emplace_back(std::make_unique<event_node>(event_node{
                            lc.d_time_,
                            get_schedule_time(sched, section.edge(),
                                              trp->lcon_idx_, event_type::DEP),
                            event_type::DEP,
                            true,
                            section.from_station_id(),
                            {},
                            {}}))
                        .get();
    auto arr_node = data.graph_.nodes_
                        .emplace_back(std::make_unique<event_node>(event_node{
                            lc.a_time_,
                            get_schedule_time(sched, section.edge(),
                                              trp->lcon_idx_, event_type::ARR),
                            event_type::ARR,
                            true,
                            section.to_station_id(),
                            {},
                            {}}))
                        .get();
    auto const capacity =
        get_capacity(sched, lc, data.trip_capacity_map_,
                     data.category_capacity_map_, data.default_capacity_);
    edges.emplace_back(add_edge(
        make_trip_edge(dep_node, arr_node, edge_type::TRIP, trp, capacity)));
    if (prev_node != nullptr) {
      add_edge(
          make_trip_edge(prev_node, dep_node, edge_type::WAIT, trp, capacity));
    }
    prev_node = arr_node;
  }
  return edges;
}

trip_data* get_or_add_trip(schedule const& sched, paxmon_data& data,
                           extern_trip const& et) {
  return utl::get_or_create(data.graph_.trip_data_, et,
                            [&]() {
                              return std::make_unique<trip_data>(
                                  trip_data{add_trip(sched, data, et), {}, {}});
                            })
      .get();
}

void add_interchange_edges(event_node const* evn,
                           std::vector<edge*>& updated_interchange_edges,
                           graph const& g, system_statistics& system_stats) {
  if (evn->type_ == event_type::ARR) {
    return utl::all(evn->outgoing_edges(g))  //
           | utl::transform([](auto&& e) { return e.get(); })  //
           | utl::remove_if([](auto&& e) {
               return e->type_ != edge_type::INTERCHANGE;
             })  //
           | utl::for_each([&](auto&& e) {
               ++system_stats.total_updated_interchange_edges_;
               updated_interchange_edges.push_back(e);
             });
  } else /*if (evn->type_ == event_type::DEP)*/ {
    assert(evn->type_ == event_type::DEP);
    return utl::all(evn->incoming_edges(g))  //
           | utl::remove_if([](auto&& e) {
               return e->type_ != edge_type::INTERCHANGE;
             })  //
           | utl::for_each([&](auto&& e) {
               ++system_stats.total_updated_interchange_edges_;
               updated_interchange_edges.push_back(e);
             });
  }
}

void update_event_times(schedule const& sched, graph& g,
                        RtDelayUpdate const* du,
                        std::vector<edge*>& updated_interchange_edges,
                        system_statistics& system_stats) {
  auto const trp = from_fbs(sched, du->trip());
  auto const et = to_extern_trip(sched, trp);
  auto trip_edges = g.trip_data_.find(et);
  if (trip_edges == end(g.trip_data_)) {
    return;
  }
  ++system_stats.update_event_times_trip_edges_found_;
  for (auto const& ue : *du->events()) {
    auto const station_id =
        get_station(sched, ue->base()->station_id()->str())->index_;
    auto const schedule_time =
        unix_to_motistime(sched, ue->base()->schedule_time());
    for (auto te : trip_edges->second->edges_) {
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
                       std::vector<edge*>& updated_interchange_edges,
                       system_statistics& system_stats) {
  ++system_stats.update_trip_route_count_;
  auto const trp = from_fbs(sched, ru->trip());
  auto const et = to_extern_trip(sched, trp);
  auto td = data.graph_.trip_data_.find(et);
  if (td == end(data.graph_.trip_data_)) {
    return;
  }
  ++system_stats.update_trip_route_trip_edges_found_;

  auto const current_teks = to_trip_ev_keys(*td->second, data.graph_);
  auto const new_teks = to_trip_ev_keys(sched, *ru->new_route());

  apply_reroute(data, sched, trp, et, *td->second, current_teks, new_teks,
                updated_interchange_edges);
}

void add_passenger_group_to_edge(edge* e, passenger_group* pg) {
  for (auto& psi : e->pax_connection_info_.section_infos_) {
    if (psi.group_ == pg) {
      if (!psi.valid_) {
        psi.valid_ = true;
        e->passengers_ += psi.group_->passengers_;
      }
      return;
    }
  }
  e->pax_connection_info_.section_infos_.emplace_back(pg);
  e->passengers_ += pg->passengers_;
}

void remove_passenger_group_from_edge(edge* e, passenger_group* pg) {
  for (auto& psi : e->pax_connection_info_.section_infos_) {
    if (psi.group_ == pg) {
      if (psi.valid_) {
        psi.valid_ = false;
        e->passengers_ -= psi.group_->passengers_;
      }
      return;
    }
  }
}

void for_each_trip(
    schedule const& sched, paxmon_data& data, compact_journey const& journey,
    std::function<void(journey_leg const&, trip_data const*)> const& fn) {
  for (auto const& leg : journey.legs_) {
    fn(leg, get_or_add_trip(sched, data, leg.trip_));
  }
}

void for_each_edge(schedule const& sched, paxmon_data& data,
                   compact_journey const& journey,
                   std::function<void(journey_leg const&, edge*)> const& fn) {
  for_each_trip(sched, data, journey,
                [&](journey_leg const& leg, trip_data const* td) {
                  auto in_trip = false;
                  for (auto e : td->edges_) {
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

}  // namespace motis::paxmon
