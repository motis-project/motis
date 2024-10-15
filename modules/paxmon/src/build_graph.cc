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
                     passenger_group* grp, duration transfer_time,
                     universe& uv) {
  for (auto& e : uv.graph_.outgoing_edges(from)) {
    if (e.type_ == edge_type::INTERCHANGE && e.to_ == to &&
        e.transfer_time() == transfer_time) {
      add_passenger_group_to_edge(uv, &e, grp);
      grp->edges_.emplace_back(get_edge_index(uv, &e));
      return;
    }
  }
  auto pci = uv.pax_connection_info_.insert();
  uv.pax_connection_info_.groups_[pci].emplace_back(grp->id_);
  uv.pax_connection_info_.init_expected_load(uv.passenger_groups_, pci);
  auto const* e =
      add_edge(uv, make_interchange_edge(from, to, transfer_time, pci));
  auto const ei = get_edge_index(uv, e);
  grp->edges_.emplace_back(ei);

  auto const from_station = uv.graph_.nodes_[from].station_idx();
  auto const to_station = uv.graph_.nodes_[to].station_idx();
  uv.interchanges_at_station_[from_station].emplace_back(ei);
  if (from_station != to_station) {
    uv.interchanges_at_station_[to_station].emplace_back(ei);
  }
}

};  // namespace

void add_passenger_group_to_graph(schedule const& sched,
                                  capacity_maps const& caps, universe& uv,
                                  passenger_group& grp) {
  utl::verify(grp.edges_.empty(), "group already added to graph");
  auto exit_node = INVALID_EVENT_NODE_INDEX;
  auto last_trip = INVALID_TRIP_DATA_INDEX;

  for (auto const& leg : grp.compact_planned_journey_.legs_) {
    utl::verify(leg.enter_time_ != INVALID_TIME, "invalid enter time");
    utl::verify(leg.exit_time_ != INVALID_TIME, "invalid exit time");

    auto tdi = INVALID_TRIP_DATA_INDEX;
    try {
      tdi = get_or_add_trip(sched, caps, uv, leg.trip_idx_);
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
    for (auto& ei : uv.trip_data_.edges(tdi)) {
      if (!in_trip) {
        auto* e = ei.get(uv);
        auto const from = e->from(uv);
        if (from->station_ == leg.enter_station_id_ &&
            from->schedule_time_ == leg.enter_time_) {
          in_trip = true;
          enter_found = true;
          if (exit_node == INVALID_EVENT_NODE_INDEX) {
            exit_node = uv.trip_data_.enter_exit_node(tdi);
          }
          auto const transfer_time = get_transfer_duration(leg.enter_transfer_);
          add_interchange(exit_node, from->index_, &grp, transfer_time, uv);
        }
      }
      if (in_trip) {
        auto* e = ei.get(uv);
        add_passenger_group_to_edge(uv, e, &grp);
        grp.edges_.emplace_back(ei);
        auto const to = e->to(uv);
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
        auto* e = ei.get(uv);
        remove_passenger_group_from_edge(uv, e, &grp);
      }
      grp.edges_.clear();

      std::cout << "add_passenger_group_to_graph: enter_found=" << enter_found
                << ", exit_found=" << exit_found << "\n";

      std::cout << "current leg:\n";
      print_leg(sched, leg);

      std::cout << "\ncurrent trip:\n";
      print_trip_sections(uv, sched, leg.trip_idx_, tdi);

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
    add_interchange(exit_node, uv.trip_data_.enter_exit_node(last_trip), &grp,
                    0, uv);
  }

  utl::verify(!grp.edges_.empty(), "empty passenger group edges");
}

void remove_passenger_group_from_graph(universe& uv, passenger_group* pg) {
  for (auto const& ei : pg->edges_) {
    auto* e = ei.get(uv);
    auto guard = std::lock_guard{uv.pax_connection_info_.mutex(e->pci_)};
    remove_passenger_group_from_edge(uv, e, pg);
  }
  pg->edges_.clear();
}

build_graph_stats build_graph_from_journeys(schedule const& sched,
                                            capacity_maps const& caps,
                                            universe& uv) {
  scoped_timer build_graph_timer{"build paxmon graph from journeys"};
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->in_high(uv.passenger_groups_.size());

  auto stats = build_graph_stats{};
  for (auto* pg : uv.passenger_groups_) {
    utl::verify(pg != nullptr, "null passenger group");
    utl::verify(!pg->compact_planned_journey_.legs_.empty(),
                "empty passenger group");
    try {
      add_passenger_group_to_graph(sched, caps, uv, *pg);
      if (pg->edges_.empty()) {
        uv.passenger_groups_.release(pg->id_);
      }
    } catch (std::system_error const& e) {
      LOG(motis::logging::error)
          << "could not add passenger group: " << e.what();
      ++stats.groups_not_added_;
    }
    progress_tracker->increment();
  }

  for (auto idx = pci_index{0}; idx < uv.pax_connection_info_.size(); ++idx) {
    uv.pax_connection_info_.init_expected_load(uv.passenger_groups_, idx);
  }

  if (stats.groups_not_added_ != 0) {
    LOG(motis::logging::error)
        << "could not add " << stats.groups_not_added_ << " passenger groups";
  }
  return stats;
}

}  // namespace motis::paxmon
