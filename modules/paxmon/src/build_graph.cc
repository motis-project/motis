#include "motis/paxmon/build_graph.h"

#include <iostream>

#include "utl/erase_if.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/paxmon/debug.h"
#include "motis/paxmon/graph_access.h"

using namespace motis::logging;

namespace motis::paxmon {

namespace {

void add_interchange(event_node* from, event_node* to, passenger_group* grp,
                     duration transfer_time, graph const& g) {
  for (auto& e : from->outgoing_edges(g)) {
    if (e->type_ == edge_type::INTERCHANGE && e->to(g) == to &&
        e->transfer_time() == transfer_time) {
      add_passenger_group_to_edge(e.get(), grp);
      grp->edges_.emplace_back(e.get());
      return;
    }
  }
  grp->edges_.emplace_back(add_edge(make_interchange_edge(
      from, to, transfer_time, pax_connection_info{grp})));
}

};  // namespace

void add_passenger_group_to_graph(schedule const& sched, paxmon_data& data,
                                  passenger_group& grp) {
  utl::verify(grp.edges_.empty(), "group already added to graph");
  event_node* exit_node = nullptr;
  trip_data* last_trip = nullptr;

  for (auto const& leg : grp.compact_planned_journey_.legs_) {
    utl::verify(leg.enter_time_ != INVALID_TIME, "invalid enter time");
    utl::verify(leg.exit_time_ != INVALID_TIME, "invalid exit time");

    trip_data* te = nullptr;
    try {
      te = get_or_add_trip(sched, data, leg.trip_);
    } catch (std::system_error const& e) {
      std::cerr << "could not add trip for passenger group " << grp.id_
                << " (source=" << grp.source_.primary_ref_ << "."
                << grp.source_.secondary_ref_ << ")" << std::endl;
      throw e;
    }
    auto in_trip = false;
    last_trip = nullptr;
    auto enter_found = false;
    auto exit_found = false;
    for (auto e : te->edges_) {
      if (!in_trip) {
        auto const from = e->from(data.graph_);
        if (from->station_ == leg.enter_station_id_ &&
            from->schedule_time_ == leg.enter_time_) {
          in_trip = true;
          enter_found = true;
          if (exit_node == nullptr) {
            exit_node = &te->enter_exit_node_;
          }
          auto const transfer_time = get_transfer_duration(leg.enter_transfer_);
          add_interchange(exit_node, from, &grp, transfer_time, data.graph_);
        }
      }
      if (in_trip) {
        add_passenger_group_to_edge(e, &grp);
        grp.edges_.emplace_back(e);
        auto const to = e->to(data.graph_);
        if (to->station_ == leg.exit_station_id_ &&
            to->schedule_time_ == leg.exit_time_) {
          exit_node = to;
          last_trip = te;
          exit_found = true;
          break;
        }
      }
    }
    if (!enter_found || !exit_found) {
      for (auto e : grp.edges_) {
        remove_passenger_group_from_edge(e, &grp);
      }
      grp.edges_.clear();

      std::cout << "add_passenger_group_to_graph: enter_found=" << enter_found
                << ", exit_found=" << exit_found << "\n";

      std::cout << "current leg:\n";
      print_leg(sched, leg);

      std::cout << "\ncurrent trip:\n";
      print_trip_sections(data.graph_, sched, leg.trip_, te);

      std::cout << "\ncompact planned journey:\n";
      for (auto const& l : grp.compact_planned_journey_.legs_) {
        print_leg(sched, l);
      }

      std::cout << "\n\n";

      throw utl::fail(
          "add_passenger_group_to_graph: trip enter/exit not found");
    }
  }

  if (exit_node != nullptr && last_trip != nullptr) {
    add_interchange(exit_node, &last_trip->enter_exit_node_, &grp, 0,
                    data.graph_);
  }

  utl::verify(!grp.edges_.empty(), "empty passenger group edges");
}

void remove_passenger_group_from_graph(passenger_group* pg) {
  for (auto e : pg->edges_) {
    auto guard = std::lock_guard{e->pax_connection_info_.mutex_};
    remove_passenger_group_from_edge(e, pg);
  }
  pg->edges_.clear();
}

build_graph_stats build_graph_from_journeys(schedule const& sched,
                                            paxmon_data& data) {
  scoped_timer build_graph_timer{"build paxmon graph from journeys"};

  auto stats = build_graph_stats{};
  for (auto& pg : data.graph_.passenger_groups_) {
    utl::verify(pg != nullptr, "null passenger group");
    utl::verify(!pg->compact_planned_journey_.legs_.empty(),
                "empty passenger group");
    try {
      add_passenger_group_to_graph(sched, data, *pg);
      if (pg->edges_.empty()) {
        data.graph_.passenger_group_allocator_.release(pg);
        pg = nullptr;
      }
    } catch (std::system_error const& e) {
      LOG(motis::logging::error)
          << "could not add passenger group: " << e.what();
      ++stats.groups_not_added_;
    }
  }
  if (stats.groups_not_added_ != 0) {
    LOG(motis::logging::error)
        << "could not add " << stats.groups_not_added_ << " passenger groups";
  }
  return stats;
}

}  // namespace motis::paxmon
