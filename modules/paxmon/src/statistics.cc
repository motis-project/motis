#include "motis/paxmon/statistics.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

graph_statistics calc_graph_statistics(schedule const& sched,
                                       paxmon_data const& data) {
  graph_statistics stats;

  stats.nodes_ = data.graph_.nodes_.size();
  std::set<std::uint32_t> stations;
  std::set<trip const*> trips;
  std::set<trip const*> trips_over_capacity;
  for (auto const& n : data.graph_.nodes_) {
    stations.insert(n->station_);
    if (n->is_canceled()) {
      ++stats.canceled_nodes_;
    }
    stats.edges_ += n->outgoing_edges(data.graph_).size();
    for (auto const& e : n->outgoing_edges(data.graph_)) {
      switch (e->type()) {
        case edge_type::TRIP: {
          ++stats.trip_edges_;
          auto const& edge_trips = e->get_trips(sched);
          trips.insert(begin(edge_trips), end(edge_trips));
          break;
        }
        case edge_type::INTERCHANGE: ++stats.interchange_edges_; break;
        case edge_type::WAIT: ++stats.wait_edges_; break;
        case edge_type::THROUGH: ++stats.through_edges_; break;
      }
      if (e->is_canceled(data.graph_)) {
        ++stats.canceled_edges_;
      } else if (e->is_trip() && e->has_capacity() &&
                 get_base_load(e->get_pax_connection_info()) > e->capacity()) {
        ++stats.edges_over_capacity_;
        auto const& edge_trips = e->get_trips(sched);
        trips_over_capacity.insert(begin(edge_trips), end(edge_trips));
      }
      if (e->is_broken()) {
        ++stats.broken_edges_;
      }
    }
  }
  stats.stations_ = stations.size();
  stats.trips_ = trips.size();
  stats.trips_over_capacity_ = trips_over_capacity.size();

  stats.passenger_groups_ = data.graph_.passenger_groups_.size();
  for (auto const& pg : data.graph_.passenger_groups_) {
    if (pg == nullptr) {
      continue;
    }
    stats.passengers_ += pg->passengers_;
    if (!pg->ok_) {
      ++stats.broken_passenger_groups_;
    }
  }

  return stats;
}

}  // namespace motis::paxmon
