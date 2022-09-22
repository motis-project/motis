#include "motis/paxmon/statistics.h"

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

graph_statistics calc_graph_statistics(schedule const& sched,
                                       universe const& uv) {
  graph_statistics stats;

  stats.nodes_ = uv.graph_.nodes_.size();
  std::set<std::uint32_t> stations;
  std::set<trip const*> trips;
  std::set<trip const*> trips_over_capacity;
  for (auto const& n : uv.graph_.nodes_) {
    stations.insert(n.station_);
    if (n.is_canceled()) {
      ++stats.canceled_nodes_;
    }
    stats.edges_ += n.outgoing_edges(uv).size();
    for (auto const& e : n.outgoing_edges(uv)) {
      switch (e.type()) {
        case edge_type::TRIP: {
          ++stats.trip_edges_;
          auto const& edge_trips = e.get_trips(sched);
          trips.insert(begin(edge_trips), end(edge_trips));
          break;
        }
        case edge_type::INTERCHANGE: ++stats.interchange_edges_; break;
        case edge_type::WAIT: ++stats.wait_edges_; break;
        case edge_type::THROUGH: ++stats.through_edges_; break;
        case edge_type::DISABLED: ++stats.disabled_edges_; break;
      }
      if (e.is_canceled(uv)) {
        ++stats.canceled_edges_;
      } else if (e.is_trip() && e.has_capacity() &&
                 get_base_load(uv.passenger_groups_,
                               uv.pax_connection_info_.group_routes(e.pci_)) >
                     e.capacity()) {
        ++stats.edges_over_capacity_;
        auto const& edge_trips = e.get_trips(sched);
        trips_over_capacity.insert(begin(edge_trips), end(edge_trips));
      }
      if (e.is_broken()) {
        ++stats.broken_edges_;
      }
    }
  }
  stats.stations_ = stations.size();
  stats.trips_ = trips.size();
  stats.trips_over_capacity_ = trips_over_capacity.size();

  stats.passenger_groups_ = uv.passenger_groups_.size();
  for (auto const& pg : uv.passenger_groups_) {
    if (pg == nullptr) {
      continue;
    }
    stats.passengers_ += pg->passengers_;
    auto routes = uv.passenger_groups_.routes(pg->id_);
    stats.passenger_group_routes_ += routes.size();
    for (auto const& gr : routes) {
      if (gr.broken_) {
        ++stats.broken_passenger_group_routes_;
      }
    }
  }

  return stats;
}

}  // namespace motis::paxmon
