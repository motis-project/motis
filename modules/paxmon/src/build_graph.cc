#include "motis/paxmon/build_graph.h"

#include <iostream>
#include <optional>

#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/paxmon/debug.h"
#include "motis/paxmon/graph_access.h"

using namespace motis::logging;

namespace motis::paxmon {

namespace {

void add_interchange(event_node_index const from, event_node_index const to,
                     passenger_group_with_route const pgwr,
                     dynamic_fws_multimap<edge_index>::mutable_bucket& edges,
                     duration const transfer_time, universe& uv,
                     schedule const& sched, bool const log,
                     pci_log_reason_t const reason) {
  std::optional<pci_index> through_pci;
  for (auto& e : uv.graph_.outgoing_edges(from)) {
    if (e.type_ == edge_type::INTERCHANGE && e.to_ == to &&
        e.transfer_time() == transfer_time) {
      add_group_route_to_edge(uv, sched, &e, pgwr, log, reason);
      edges.emplace_back(get_edge_index(uv, &e));
      return;
    } else if (e.type_ == edge_type::THROUGH && e.to_ == to) {
      through_pci = e.pci_;
    }
  }
  auto pci = through_pci ? *through_pci : uv.pax_connection_info_.insert();
  uv.pax_connection_info_.group_routes_[pci].emplace_back(pgwr);
  uv.pax_connection_info_.init_expected_load(uv.passenger_groups_, pci);
  auto const* e =
      add_edge(uv, make_interchange_edge(from, to, transfer_time, pci));
  auto const ei = get_edge_index(uv, e);
  edges.emplace_back(ei);

  if (log && uv.graph_log_.enabled_) {
    uv.graph_log_.pci_log_[pci].emplace_back(pci_log_entry{
        sched.system_time_, pci_log_action_t::ROUTE_ADDED, reason, pgwr});
  }

  auto const from_station = uv.graph_.nodes_[from].station_idx();
  auto const to_station = uv.graph_.nodes_[to].station_idx();
  uv.interchanges_at_station_[from_station].emplace_back(ei);
  if (from_station != to_station) {
    uv.interchanges_at_station_[to_station].emplace_back(ei);
  }
}

};  // namespace

add_group_route_to_graph_result add_group_route_to_graph(
    schedule const& sched, capacity_maps const& caps, universe& uv,
    passenger_group const& grp, group_route const& gr, bool const log,
    pci_log_reason_t const reason) {
  auto result = add_group_route_to_graph_result{};
  auto edges = uv.passenger_groups_.route_edges(gr.edges_index_);
  auto const cj = uv.passenger_groups_.journey(gr.compact_journey_index_);
  auto const pgwr =
      passenger_group_with_route{grp.id_, gr.local_group_route_index_};

  utl::verify(edges.empty(), "group already added to graph");
  auto exit_node = INVALID_EVENT_NODE_INDEX;
  auto last_trip = INVALID_TRIP_DATA_INDEX;

  for (auto const& leg : cj.legs()) {
    utl::verify(leg.enter_time_ != INVALID_TIME, "invalid enter time");
    utl::verify(leg.exit_time_ != INVALID_TIME, "invalid exit time");

    auto tdi = INVALID_TRIP_DATA_INDEX;
    try {
      tdi = get_or_add_trip(sched, caps, uv, leg.trip_idx_);
    } catch (std::system_error const& e) {
      std::cerr << "could not add trip for passenger group " << grp.id_
                << " (source=" << grp.source_.primary_ref_ << "."
                << grp.source_.secondary_ref_ << "), route "
                << gr.local_group_route_index_ << std::endl;
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
          add_interchange(exit_node, from->index_, pgwr, edges, transfer_time,
                          uv, sched, log, reason);
        }
      }
      if (in_trip) {
        auto* e = ei.get(uv);
        add_group_route_to_edge(uv, sched, e, pgwr, log, reason);
        edges.emplace_back(ei);
        auto const to = e->to(uv);
        if (to->station_ == leg.exit_station_id_ &&
            to->schedule_time_ == leg.exit_time_) {
          exit_node = to->index_;
          last_trip = tdi;
          exit_found = true;
          result.scheduled_arrival_time_ = to->schedule_time_;
          result.current_arrival_time_ = to->time_;
          break;
        }
      }
    }
    if (!enter_found || !exit_found) {
      for (auto const& ei : edges) {
        auto* e = ei.get(uv);
        remove_group_route_from_edge(uv, sched, e, pgwr, log, reason);
      }
      edges.clear();

      std::cout << "add_group_route_to_graph: enter_found=" << enter_found
                << ", exit_found=" << exit_found << "\n";

      std::cout << "current leg:\n";
      print_leg(sched, leg);

      std::cout << "\ncurrent trip:\n";
      print_trip_sections(uv, sched, leg.trip_idx_, tdi);

      std::cout << "\ncompact planned journey:\n";
      for (auto const& l : cj.legs()) {
        print_leg(sched, l);
      }

      std::cout << "\n\n";

      throw utl::fail("add_group_route_to_graph: trip enter/exit not found");
    }
  }

  if (exit_node != INVALID_EVENT_NODE_INDEX &&
      last_trip != INVALID_TRIP_DATA_INDEX) {
    add_interchange(exit_node, uv.trip_data_.enter_exit_node(last_trip), pgwr,
                    edges, 0, uv, sched, log, reason);
  }

  utl::verify(!edges.empty(), "empty passenger group edges");
  return result;
}

void remove_group_route_from_graph(universe& uv, schedule const& sched,
                                   passenger_group const& grp,
                                   group_route const& gr, bool const log,
                                   pci_log_reason_t reason) {
  auto edges = uv.passenger_groups_.route_edges(gr.edges_index_);
  auto const pgwr =
      passenger_group_with_route{grp.id_, gr.local_group_route_index_};
  for (auto const& ei : edges) {
    auto* e = ei.get(uv);
    auto guard = std::lock_guard{uv.pax_connection_info_.mutex(e->pci_)};
    remove_group_route_from_edge(uv, sched, e, pgwr, log, reason);
  }
  edges.clear();
}

}  // namespace motis::paxmon
