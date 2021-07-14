#include "motis/paxmon/build_graph.h"

#include <iostream>

#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/paxmon/debug.h"
#include "motis/paxmon/graph_access.h"

using namespace motis::logging;

namespace motis::paxmon {

namespace {

void add_interchange(event_node_index from, event_node_index to,
                     passenger_group* grp, duration transfer_time, graph& g) {
  for (auto& e : g.graph_.outgoing_edges(from)) {
    if (e.type_ == edge_type::INTERCHANGE && e.to_ == to &&
        e.transfer_time() == transfer_time) {
      add_passenger_group_to_edge(&e, grp);
      grp->edges_.emplace_back(get_edge_index(g, &e));
      return;
    }
  }
  auto pci = pax_connection_info{grp->id_};
  pci.init_expected_load(g.passenger_groups_);
  auto const* e = add_edge(
      g, make_interchange_edge(from, to, transfer_time, std::move(pci)));
  auto const ei = get_edge_index(g, e);
  grp->edges_.emplace_back(ei);
}

};  // namespace

void add_passenger_group_to_graph(schedule const& sched, paxmon_data& data,
                                  passenger_group& grp) {
  utl::verify(grp.edges_.empty(), "group already added to graph");
  auto exit_node = INVALID_EVENT_NODE_INDEX;
  auto last_trip = INVALID_TRIP_DATA_INDEX;

  for (auto const& leg : grp.compact_planned_journey_.legs_) {
    utl::verify(leg.enter_time_ != INVALID_TIME, "invalid enter time");
    utl::verify(leg.exit_time_ != INVALID_TIME, "invalid exit time");

    auto tdi = INVALID_TRIP_DATA_INDEX;
    try {
      tdi = get_or_add_trip(sched, data, leg.trip_);
    } catch (std::system_error const& e) {
      std::cerr << "could not add trip for passenger group " << grp.id_
                << " (source=" << grp.source_.primary_ref_ << "."
                << grp.source_.secondary_ref_ << ")" << std::endl;
      throw e;
    }
    auto in_trip = false;
    last_trip = INVALID_TRIP_DATA_INDEX;
    auto enter_found = false;
    auto exit_found = false;
    for (auto& ei : data.graph_.trip_data_.edges(tdi)) {
      if (!in_trip) {
        auto* e = ei.get(data.graph_);
        auto const from = e->from(data.graph_);
        if (from->station_ == leg.enter_station_id_ &&
            from->schedule_time_ == leg.enter_time_) {
          in_trip = true;
          enter_found = true;
          if (exit_node == INVALID_EVENT_NODE_INDEX) {
            exit_node = data.graph_.trip_data_.enter_exit_node(tdi);
          }
          auto const transfer_time = get_transfer_duration(leg.enter_transfer_);
          add_interchange(exit_node, from->index_, &grp, transfer_time,
                          data.graph_);
        }
      }
      if (in_trip) {
        add_passenger_group_to_edge(e, &grp);
        auto* e = ei.get(data.graph_);
        grp.edges_.emplace_back(ei);
        auto const to = e->to(data.graph_);
        if (to->station_ == leg.exit_station_id_ &&
            to->schedule_time_ == leg.exit_time_) {
          exit_node = to->index_;
          last_trip = tdi;
          exit_found = true;
          break;
        }
      }
    }
    if (!enter_found || !exit_found) {
      for (auto const& ei : grp.edges_) {
        auto* e = ei.get(data.graph_);
        remove_passenger_group_from_edge(e, &grp);
      }
      grp.edges_.clear();

      std::cout << "add_passenger_group_to_graph: enter_found=" << enter_found
                << ", exit_found=" << exit_found << "\n";

      std::cout << "current leg:\n";
      print_leg(sched, leg);

      std::cout << "\ncurrent trip:\n";
      print_trip_sections(data.graph_, sched, leg.trip_, tdi);

      std::cout << "\ncompact planned journey:\n";
      for (auto const& l : grp.compact_planned_journey_.legs_) {
        print_leg(sched, l);
      }

      std::cout << "\n\n";

      throw utl::fail(
          "add_passenger_group_to_graph: trip enter/exit not found");
    }
  }

  if (exit_node != INVALID_EVENT_NODE_INDEX &&
      last_trip != INVALID_TRIP_DATA_INDEX) {
    add_interchange(exit_node,
                    data.graph_.trip_data_.enter_exit_node(last_trip), &grp, 0,
                    data.graph_);
  }

  utl::verify(!grp.edges_.empty(), "empty passenger group edges");
}

void remove_passenger_group_from_graph(paxmon_data& data, passenger_group* pg) {
  for (auto const& ei : pg->edges_) {
    auto* e = ei.get(data.graph_);
    auto guard = std::lock_guard{e->get_pax_connection_info().mutex_};
    remove_passenger_group_from_edge(e, pg);
  }
  pg->edges_.clear();
}

build_graph_stats build_graph_from_journeys(schedule const& sched,
                                            paxmon_data& data) {
  scoped_timer build_graph_timer{"build paxmon graph from journeys"};
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->in_high(data.graph_.passenger_groups_.size());

  auto stats = build_graph_stats{};
  for (auto* pg : data.graph_.passenger_groups_) {
    utl::verify(pg != nullptr, "null passenger group");
    utl::verify(!pg->compact_planned_journey_.legs_.empty(),
                "empty passenger group");
    try {
      add_passenger_group_to_graph(sched, data, *pg);
      if (pg->edges_.empty()) {
        data.graph_.passenger_groups_.release(pg->id_);
      }
    } catch (std::system_error const& e) {
      LOG(motis::logging::error)
          << "could not add passenger group: " << e.what();
      ++stats.groups_not_added_;
    }
    progress_tracker->increment();
  }
  if (stats.groups_not_added_ != 0) {
    LOG(motis::logging::error)
        << "could not add " << stats.groups_not_added_ << " passenger groups";
  }
  return stats;
}

}  // namespace motis::paxmon
