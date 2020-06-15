#include "motis/rsl/graph_access.h"

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

#include "motis/rsl/capacity.h"
#include "motis/rsl/reroute.h"

namespace motis::rsl {

using namespace motis::rt;

std::vector<edge*> add_trip(schedule const& sched, rsl_data& data,
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
    auto const capacity = get_capacity(sched, data.capacity_map_, lc);
    edges.emplace_back(add_edge(edge{
        dep_node, arr_node, edge_type::TRIP, trp, 0, capacity, 0, false, {}}));
    if (prev_node != nullptr) {
      add_edge(edge{prev_node,
                    dep_node,
                    edge_type::WAIT,
                    trp,
                    0,
                    capacity,
                    0,
                    false,
                    {}});
    }
    prev_node = arr_node;
  }
  return edges;
}

trip_data* get_or_add_trip(schedule const& sched, rsl_data& data,
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
                           system_statistics& system_stats) {
  if (evn->type_ == event_type::ARR) {
    return utl::all(evn->out_edges_)  //
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
    return utl::all(evn->in_edges_)  //
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
      if (ue->base()->event_type() == EventType_DEP &&
          te->from_->type_ == event_type::DEP &&
          te->from_->station_ == station_id &&
          te->from_->schedule_time_ == schedule_time) {
        ++system_stats.update_event_times_dep_updated_;
        te->from_->time_ =
            unix_to_motistime(sched.schedule_begin_, ue->updated_time());
        add_interchange_edges(te->from_, updated_interchange_edges,
                              system_stats);
      } else if (ue->base()->event_type() == EventType_ARR &&
                 te->to_->type_ == event_type::ARR &&
                 te->to_->station_ == station_id &&
                 te->to_->schedule_time_ == schedule_time) {
        ++system_stats.update_event_times_arr_updated_;
        te->to_->time_ =
            unix_to_motistime(sched.schedule_begin_, ue->updated_time());
        add_interchange_edges(te->to_, updated_interchange_edges, system_stats);
      }
    }
  }
}

void update_trip_route(schedule const& sched, rsl_data& data,
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

  /*
  std::cout << "### begin reroute ###\n"
            << "  trip: {station_id=" << et.station_id_
            << ", train_nr=" << et.train_nr_
            << ", time=" << format_unixtime(et.time_)
            << ", target_station_id=" << et.target_station_id_
            << ", target_time=" << format_unixtime(et.target_time_)
            << ", line_id=" << et.line_id_
            << "}\n  old route: " << ru->old_route()->size() << " events\n";

  auto const print_event = [](auto const& ei) {
    std::cout << "    station_id=" << ei->station_id()->str()
              << ", schedule_time=" << format_unixtime(ei->schedule_time())
              << ", event_type="
              << (ei->event_type() == EventType_DEP ? "DEP" : "ARR") << "\n";
  };
  for (auto const& ei : *ru->old_route()) {
    print_event(ei);
  }

  std::cout << "  new route: " << ru->new_route()->size() << " events\n";

  for (auto const& ei : *ru->new_route()) {
    print_event(ei);
  }
  std::cout << "  current graph route:\n";
  auto const print_node = [&](event_node const* en) {
    std::cout << "    station_id=" << sched.stations_[en->station_]->eva_nr_
              << ", schedule_time=" << format_time(en->schedule_time_)
              << ", time=" << format_time(en->time_)
              << ", type=" << (en->type_ == event_type::DEP ? "DEP" : "ARR")
              << ", valid=" << en->valid_ << "\n";
  };
  for (auto const te : td->second->edges_) {
    print_node(te->from_);
    std::cout << "    | " << te->type_ << " passengers=" << te->passengers_
              << ", groups=" << te->rsl_connection_info_.section_infos_.size()
              << "\n";
    print_node(te->to_);
    std::cout << "\n";
  }

  std::set<passenger_group*> affected_passenger_groups;
  for (auto const te : td->second->edges_) {
    for (auto const& psi : te->rsl_connection_info_.section_infos_) {
      if (psi.valid_) {
        affected_passenger_groups.insert(psi.group_);
      }
    }
  }
  std::cout << affected_passenger_groups.size()
            << " affected passenger groups\n";
  for (auto const pg : affected_passenger_groups) {
    std::cout << "passenger group with " << pg->passengers_
              << " passengers and " << pg->compact_planned_journey_.legs_.size()
              << " legs:\n";
    for (auto const& leg : pg->compact_planned_journey_.legs_) {
      if (leg.trip_ == et) {
        auto const& enter_station = sched.stations_[leg.enter_station_id_];
        auto const& enter_eva = enter_station->eva_nr_;
        auto const& exit_station = sched.stations_[leg.exit_station_id_];
        auto const& exit_eva = exit_station->eva_nr_;
        auto const enter_ut = motis_to_unixtime(sched, leg.enter_time_);
        auto const exit_ut = motis_to_unixtime(sched, leg.exit_time_);
        std::cout << "  trip: enter=" << enter_eva << "/"
                  << enter_station->index_ << " @ "
                  << format_time(leg.enter_time_) << ", exit=" << exit_eva
                  << "/" << exit_station->index_ << " @ "
                  << format_time(leg.exit_time_);
        if (leg.enter_transfer_) {
          std::cout << ", enter_transfer=" << leg.enter_transfer_->duration_
                    << "min "
                    << (leg.enter_transfer_->type_ ==
                                transfer_info::type::FOOTPATH
                            ? "footpath"
                            : "interchange");
        }
        std::cout << "\n";

        auto enter_found = false;
        auto enter_same_time = false;
        auto exit_found = false;
        auto exit_same_time = false;

        for (auto const& ei : *ru->new_route()) {
          auto event_eva = std::string_view{ei->station_id()->c_str(),
                                            ei->station_id()->size()};
          if (event_eva == enter_eva) {
            std::cout << "    enter station found in new route: schedule_time="
                      << format_unixtime(ei->schedule_time()) << ", event_type="
                      << (ei->event_type() == EventType_DEP ? "DEP" : "ARR")
                      << " [original time: " << format_unixtime(enter_ut)
                      << "]\n";
            if (ei->event_type() == EventType_DEP) {
              enter_found = true;
              if (ei->schedule_time() == enter_ut) {
                enter_same_time = true;
              }
            }
          } else if (event_eva == exit_eva) {
            std::cout << "    exit station found in new route: schedule_time="
                      << format_unixtime(ei->schedule_time()) << ", event_type="
                      << (ei->event_type() == EventType_DEP ? "DEP" : "ARR")
                      << " [original time: " << format_unixtime(exit_ut)
                      << "]\n";
            if (ei->event_type() == EventType_ARR) {
              exit_found = true;
              if (ei->schedule_time() == exit_ut) {
                exit_same_time = true;
              }
            }
          }
        }
        std::cout << "    enter_found=" << enter_found
                  << " [same_time=" << enter_same_time
                  << "], exit_found=" << exit_found
                  << " [same_time=" << exit_same_time << "] "
                  << (enter_found && exit_found
                          ? (enter_same_time && exit_same_time
                                 ? "OK"
                                 : "TIMES CHANGED")
                          : "BROKEN")
                  << " ***\n";
      }
    }
  }
  */

  auto const current_teks = to_trip_ev_keys(*td->second);
  auto const new_teks = to_trip_ev_keys(sched, *ru->new_route());

  /*
  std::cout << "=============================================\n"
            << "Current Events:\n";
  for (auto const& tek : current_teks) {
    std::cout << "  " << tek << "\n";
  }
  std::cout << "New Events:\n";
  for (auto const& tek : new_teks) {
    std::cout << "  " << tek << "\n";
  }
   */

  apply_reroute(data, sched, trp, et, *td->second, current_teks, new_teks,
                updated_interchange_edges);

  // std::cout << "### end reroute ###\n";
}

void add_passenger_group_to_edge(edge* e, passenger_group* pg) {
  for (auto& psi : e->rsl_connection_info_.section_infos_) {
    if (psi.group_ == pg) {
      if (!psi.valid_) {
        psi.valid_ = true;
        e->passengers_ += psi.group_->passengers_;
      }
      return;
    }
  }
  e->rsl_connection_info_.section_infos_.emplace_back(pg);
  e->passengers_ += pg->passengers_;
}

void remove_passenger_group_from_edge(edge* e, passenger_group* pg) {
  for (auto& psi : e->rsl_connection_info_.section_infos_) {
    if (psi.group_ == pg) {
      if (psi.valid_) {
        psi.valid_ = false;
        e->passengers_ -= psi.group_->passengers_;
      }
      return;
    }
  }
}

}  // namespace motis::rsl
