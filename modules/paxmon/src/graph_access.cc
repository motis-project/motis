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

#include "motis/core/common/logging.h"
#include "motis/core/debug/fbs.h"
#include "motis/core/debug/trip.h"

#include "motis/module/message.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/checks.h"
#include "motis/paxmon/graph_index.h"
#include "motis/paxmon/reroute.h"
#include "motis/paxmon/update_trip_capacity_status.h"

namespace motis::paxmon {

using namespace motis::rt;

struct rule_trip_adder {
  using rule_node_key =
      std::tuple<uint32_t /* station_idx */, time /* schedule_time */,
                 uint32_t /* merged_trips_idx */>;

  rule_trip_adder(schedule const& sched, universe& uv)
      : sched_{sched}, uv_{uv} {}

  trip_data_index add_trip(trip const* trp) {
    if (auto const [_, inserted] = trips_.insert(trp); !inserted) {
      return INVALID_TRIP_DATA_INDEX;
    }

    utl::verify(!uv_.trip_data_.contains(trp->trip_idx_),
                "trip data already exists");

    auto const enter_exit_node_idx =
        static_cast<event_node_index>(uv_.graph_.nodes_.size());
    uv_.graph_.emplace_back_node(enter_exit_node_idx);

    auto const tdi =
        uv_.trip_data_.insert_trip(trp->trip_idx_, enter_exit_node_idx);
    auto trip_edges = uv_.trip_data_.edges(tdi);

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

    update_trip_capacity_status(sched_, uv_, trp, tdi);

    return tdi;
  }

  trip_data_index get_or_add_trip(trip const* trp) {
    if (auto td = add_trip(trp); td != INVALID_TRIP_DATA_INDEX) {
      return td;
    } else {
      return uv_.trip_data_.get_index(trp->trip_idx_);
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
    auto const lcon_idx = section.trip_->lcon_idx_;

    auto const handle_through_route_edge = [&](::motis::edge const* re,
                                               bool const incoming) {
      if (re->empty()) {
        return;
      }
      auto const& lcon = re->m_.route_edge_.conns_.at(lcon_idx);
      for (auto const& merged_trp : *sched_.merged_trips_.at(lcon.trips_)) {
        add_trip(merged_trp);
      }
      if (incoming) {
        auto const feeder_arr_node = arr_nodes_.at(rule_node_key{
            re->to_->get_station()->id_,
            get_schedule_time(sched_, re, lcon_idx, event_type::ARR),
            lcon.trips_});
        get_or_create_through_edge(feeder_arr_node, dep_node);
      } else {
        auto const connecting_dep_node = dep_nodes_.at(rule_node_key{
            re->from_->get_station()->id_,
            get_schedule_time(sched_, re, lcon_idx, event_type::DEP),
            lcon.trips_});
        get_or_create_through_edge(arr_node, connecting_dep_node);
      }
    };

    for (auto const& te : section.from_node()->incoming_edges_) {
      if (te->type() == ::motis::edge::THROUGH_EDGE) {
        for (auto const& re : te->from_->incoming_edges_) {
          handle_through_route_edge(re, true);
        }
      }
    }
    for (auto const& te : section.to_node()->edges_) {
      if (te.type() == ::motis::edge::THROUGH_EDGE) {
        for (auto const& re : te.to_->edges_) {
          handle_through_route_edge(&re, false);
        }
      }
    }
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
              static_cast<event_node_index>(uv_.graph_.nodes_.size());
          uv_.graph_.emplace_back_node(idx, section.lcon().d_time_,
                                       schedule_time, event_type::DEP, true,
                                       station_idx);
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
              static_cast<event_node_index>(uv_.graph_.nodes_.size());
          uv_.graph_.emplace_back_node(idx, section.lcon().a_time_,
                                       schedule_time, event_type::ARR, true,
                                       station_idx);
          return idx;
        });
  }

  edge_index get_or_create_trip_edge(motis::access::trip_section const& section,
                                     event_node_index dep_node,
                                     event_node_index arr_node) {
    return utl::get_or_create(trip_edges_, &section.lcon(), [&]() {
      auto const sec_cap =
          get_capacity(sched_, section.lcon(), section.ev_key_from(),
                       section.ev_key_to(), uv_.capacity_maps_);
      auto const* e = add_edge(
          uv_, make_trip_edge(uv_, dep_node, arr_node, edge_type::TRIP,
                              section.lcon().trips_, sec_cap.capacity_.seats(),
                              sec_cap.source_, section.fcon().clasz_));
      return get_edge_index(uv_, e);
    });
  }

  edge_index get_or_create_wait_edge(motis::access::trip_section const& section,
                                     event_node_index prev_node,
                                     event_node_index dep_node) {
    return utl::get_or_create(
        wait_edges_, std::make_pair(prev_node, dep_node), [&]() {
          auto const* e = add_edge(
              uv_, make_trip_edge(uv_, prev_node, dep_node, edge_type::WAIT,
                                  section.lcon().trips_, UNLIMITED_CAPACITY,
                                  capacity_source::UNLIMITED,
                                  section.fcon().clasz_));
          return get_edge_index(uv_, e);
        });
  }

  edge_index get_or_create_through_edge(event_node_index const arr_node,
                                        event_node_index const dep_node) {
    return utl::get_or_create(
        through_edges_, std::make_pair(arr_node, dep_node), [&]() {
          auto const* e =
              add_edge(uv_, make_through_edge(uv_, arr_node, dep_node));
          return get_edge_index(uv_, e);
        });
  }

  schedule const& sched_;
  universe& uv_;
  std::set<trip const*> trips_;
  std::map<rule_node_key, event_node_index> dep_nodes_;
  std::map<rule_node_key, event_node_index> arr_nodes_;
  std::map<motis::light_connection const*, edge_index> trip_edges_;
  std::map<std::pair<event_node_index, event_node_index>, edge_index>
      wait_edges_;
  std::map<std::pair<event_node_index, event_node_index>, edge_index>
      through_edges_;
};

trip_data_index add_trip(schedule const& sched, universe& uv, trip const* trp) {
  auto adder = rule_trip_adder{sched, uv};
  return adder.add_trip(trp);
}

trip_data_index get_or_add_trip(schedule const& sched, universe& uv,
                                trip_idx_t const trip_idx) {
  if (auto const idx = uv.trip_data_.find_index(trip_idx);
      idx != INVALID_TRIP_DATA_INDEX) {
    return idx;
  } else {
    return add_trip(sched, uv, get_trip(sched, trip_idx));
  }
}

trip_data_index get_or_add_trip(schedule const& sched, universe& uv,
                                trip const* trp) {
  if (auto const idx = uv.trip_data_.find_index(trp->trip_idx_);
      idx != INVALID_TRIP_DATA_INDEX) {
    return idx;
  } else {
    return add_trip(sched, uv, trp);
  }
}

trip_data_index get_or_add_trip(schedule const& sched, universe& uv,
                                extern_trip const& et) {
  return get_or_add_trip(sched, uv, get_trip(sched, et));
}

trip_data_index get_trip(universe const& uv, trip_idx_t const trip_idx) {
  return uv.trip_data_.find_index(trip_idx);
}

void add_interchange_edges(event_node* evn,
                           std::vector<edge_index>& updated_interchange_edges,
                           universe& uv) {
  if (evn->type_ == event_type::ARR) {
    auto oe = evn->outgoing_edges(uv);
    return utl::all(oe)  //
           | utl::remove_if(
                 [](auto&& e) { return e.type_ != edge_type::INTERCHANGE; })  //
           | utl::for_each([&](auto&& e) {
               ++uv.system_stats_.total_updated_interchange_edges_;
               updated_interchange_edges.push_back(get_edge_index(uv, &e));
             });
  } else /*if (evn->type_ == event_type::DEP)*/ {
    assert(evn->type_ == event_type::DEP);
    return utl::all(evn->incoming_edges(uv))  //
           | utl::remove_if(
                 [](auto&& e) { return e.type_ != edge_type::INTERCHANGE; })  //
           | utl::for_each([&](auto&& e) {
               ++uv.system_stats_.total_updated_interchange_edges_;
               updated_interchange_edges.push_back(get_edge_index(uv, &e));
             });
  }
}

int update_event_times(schedule const& sched, universe& uv,
                       RtDelayUpdate const* du,
                       std::vector<edge_index>& updated_interchange_edges) {
  using namespace motis::logging;
  auto const trp = from_fbs(sched, du->trip());
  auto const tdi = uv.trip_data_.find_index(trp->trip_idx_);
  if (tdi == INVALID_TRIP_DATA_INDEX) {
    return -1;
  }
  auto updated = 0;
  auto trip_edges = uv.trip_data_.edges(tdi);
  ++uv.system_stats_.update_event_times_trip_edges_found_;
  for (auto const& ue : *du->events()) {
    auto const station_id =
        get_station(sched, ue->base()->station_id()->str())->index_;
    auto const schedule_time =
        unix_to_motistime(sched, ue->base()->schedule_time());
    auto const new_time =
        unix_to_motistime(sched.schedule_begin_, ue->updated_time());
    for (auto tei : trip_edges) {
      auto const* te = tei.get(uv);
      auto const from = te->from(uv);
      auto const to = te->to(uv);
      if (ue->base()->event_type() == EventType_DEP &&
          from->type_ == event_type::DEP && from->station_ == station_id &&
          from->schedule_time_ == schedule_time) {
        if (from->time_ != new_time) {
          ++uv.system_stats_.update_event_times_dep_updated_;
          ++updated;
          from->time_ = new_time;
          add_interchange_edges(from, updated_interchange_edges, uv);
          if (uv.graph_log_.enabled_) {
            uv.graph_log_.node_log_[from->index_].emplace_back(
                node_log_entry{sched.system_time_, from->time_, from->valid_});
          }
        }
        break;
      } else if (ue->base()->event_type() == EventType_ARR &&
                 to->type_ == event_type::ARR && to->station_ == station_id &&
                 to->schedule_time_ == schedule_time) {
        if (to->time_ != new_time) {
          ++uv.system_stats_.update_event_times_arr_updated_;
          ++updated;
          to->time_ = new_time;
          add_interchange_edges(to, updated_interchange_edges, uv);
          if (uv.graph_log_.enabled_) {
            uv.graph_log_.node_log_[to->index_].emplace_back(
                node_log_entry{sched.system_time_, to->time_, to->valid_});
          }
        }
        break;
      }
    }
  }
  return updated;
}

void update_trip_route(schedule const& sched, universe& uv,
                       RtRerouteUpdate const* ru,
                       std::vector<edge_index>& updated_interchange_edges) {
  ++uv.system_stats_.update_trip_route_count_;
  auto const trp = from_fbs(sched, ru->trip());
  auto const tdi = uv.trip_data_.find_index(trp->trip_idx_);
  if (tdi == INVALID_TRIP_DATA_INDEX) {
    return;
  }
  ++uv.system_stats_.update_trip_route_trip_edges_found_;
  uv.update_tracker_.before_trip_rerouted(trp);

  auto const current_teks = to_trip_ev_keys(tdi, uv);
  auto const new_teks = to_trip_ev_keys(sched, *ru->new_route());

  apply_reroute(uv, sched, trp, tdi, current_teks, new_teks,
                updated_interchange_edges);
}

bool add_group_route_to_edge(universe& uv, schedule const& sched, edge* e,
                             passenger_group_with_route const& entry,
                             bool const log, pci_log_reason_t const reason) {
  auto group_routes = uv.pax_connection_info_.group_routes_[e->pci_];
  auto it = std::lower_bound(begin(group_routes), end(group_routes), entry);
  if (it == end(group_routes) || *it != entry) {
    group_routes.insert(it, entry);
    if (log && uv.graph_log_.enabled_) {
      uv.graph_log_.pci_log_[e->pci_].emplace_back(pci_log_entry{
          sched.system_time_, pci_log_action_t::ROUTE_ADDED, reason, entry});
    }
    return true;
  }
  return false;
}

bool remove_group_route_from_edge(universe& uv, schedule const& sched, edge* e,
                                  passenger_group_with_route const& entry,
                                  bool const log,
                                  pci_log_reason_t const reason) {
  auto group_routes = uv.pax_connection_info_.group_routes_[e->pci_];
  auto it = std::lower_bound(begin(group_routes), end(group_routes), entry);
  if (it != end(group_routes) && *it == entry) {
    group_routes.erase(it);
    if (log && uv.graph_log_.enabled_) {
      uv.graph_log_.pci_log_[e->pci_].emplace_back(pci_log_entry{
          sched.system_time_, pci_log_action_t::ROUTE_REMOVED, reason, entry});
    }
    return true;
  }
  return false;
}

bool add_broken_group_route_to_edge(universe& uv, schedule const& sched,
                                    edge* e,
                                    passenger_group_with_route const& entry,
                                    bool const log,
                                    pci_log_reason_t const reason) {
  auto broken_group_routes =
      uv.pax_connection_info_.broken_group_routes_[e->pci_];
  auto it = std::lower_bound(begin(broken_group_routes),
                             end(broken_group_routes), entry);
  if (it == end(broken_group_routes) || *it != entry) {
    broken_group_routes.insert(it, entry);
    if (log && uv.graph_log_.enabled_) {
      uv.graph_log_.pci_log_[e->pci_].emplace_back(
          pci_log_entry{sched.system_time_,
                        pci_log_action_t::BROKEN_ROUTE_ADDED, reason, entry});
    }
    return true;
  }
  return false;
}

bool remove_broken_group_route_from_edge(
    universe& uv, schedule const& sched, edge* e,
    passenger_group_with_route const& entry, bool const log,
    pci_log_reason_t const reason) {
  auto broken_group_routes =
      uv.pax_connection_info_.broken_group_routes_[e->pci_];
  auto it = std::lower_bound(begin(broken_group_routes),
                             end(broken_group_routes), entry);
  if (it != end(broken_group_routes) && *it == entry) {
    broken_group_routes.erase(it);
    if (log && uv.graph_log_.enabled_) {
      uv.graph_log_.pci_log_[e->pci_].emplace_back(
          pci_log_entry{sched.system_time_,
                        pci_log_action_t::BROKEN_ROUTE_REMOVED, reason, entry});
    }
    return true;
  }
  return false;
}

void for_each_trip(
    schedule const& sched, universe& uv, compact_journey const& journey,
    std::function<void(journey_leg const&, trip_data_index)> const& fn) {
  for (auto const& leg : journey.legs_) {
    fn(leg, get_or_add_trip(sched, uv, leg.trip_idx_));
  }
}

void for_each_edge(schedule const& sched, universe& uv,
                   compact_journey const& journey,
                   std::function<void(journey_leg const&, edge*)> const& fn) {
  for_each_trip(sched, uv, journey,
                [&](journey_leg const& leg, trip_data_index const tdi) {
                  auto in_trip = false;
                  for (auto const& ei : uv.trip_data_.edges(tdi)) {
                    auto* e = ei.get(uv);
                    if (!in_trip) {
                      auto const from = e->from(uv);
                      if (from->station_idx() == leg.enter_station_id_ &&
                          from->schedule_time() == leg.enter_time_) {
                        in_trip = true;
                      }
                    }
                    if (in_trip) {
                      fn(leg, e);
                      auto const to = e->to(uv);
                      if (to->station_idx() == leg.exit_station_id_ &&
                          to->schedule_time() == leg.exit_time_) {
                        break;
                      }
                    }
                  }
                });
}

event_node* find_event_node(universe& uv, trip_data_index const tdi,
                            std::uint32_t const station_idx,
                            event_type const et, time const schedule_time) {
  for (auto& ei : uv.trip_data_.edges(tdi)) {
    auto* e = ei.get(uv);
    if (et == event_type::DEP) {
      auto* n = e->from(uv);
      if (n->station_idx() == station_idx &&
          n->schedule_time() == schedule_time) {
        return n;
      }
    } else if (et == event_type::ARR) {
      auto* n = e->to(uv);
      if (n->station_idx() == station_idx &&
          n->schedule_time() == schedule_time) {
        return n;
      }
    }
  }
  return nullptr;
}

}  // namespace motis::paxmon
