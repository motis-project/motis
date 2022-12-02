#include "motis/paxmon/print_stats.h"

#include "fmt/format.h"

#include "motis/core/common/logging.h"

using namespace motis::logging;

namespace motis::paxmon {

void print_graph_stats(graph_statistics const& graph_stats) {
  LOG(info) << fmt::format(
      "{:L} passenger groups, {:L} passengers, {:L} group routes",
      graph_stats.passenger_groups_, graph_stats.passengers_,
      graph_stats.passenger_group_routes_);
  LOG(info) << fmt::format("{:L} graph nodes ({:L} canceled)",
                           graph_stats.nodes_, graph_stats.canceled_nodes_);
  LOG(info) << fmt::format(
      "{:L} graph edges ({:L} canceled): {:L} trip + {:L} interchange + {:L} "
      "wait + {:L} through + {:L} disabled",
      graph_stats.edges_, graph_stats.canceled_edges_, graph_stats.trip_edges_,
      graph_stats.interchange_edges_, graph_stats.wait_edges_,
      graph_stats.through_edges_, graph_stats.disabled_edges_);
  LOG(info) << fmt::format("{:L} stations", graph_stats.stations_);
  LOG(info) << fmt::format("{:L} trips", graph_stats.trips_);
  LOG(info) << fmt::format("over capacity: {:L} trips, {:L} edges",
                           graph_stats.trips_over_capacity_,
                           graph_stats.edges_over_capacity_);
  LOG(info) << fmt::format("broken: {:L} interchange edges, {:L} group routes",
                           graph_stats.broken_edges_,
                           graph_stats.broken_passenger_group_routes_);
}

void print_allocator_stats(universe const& uv) {
  auto const& allocator = uv.passenger_groups_.allocator_;
  LOG(info) << fmt::format(
      "passenger group allocator: {:L} groups, {:.2f} MiB currently allocated, "
      "{:L} free list entries, {:L} total allocations, {:L} total "
      "deallocations",
      allocator.elements_allocated(),
      static_cast<double>(allocator.bytes_allocated()) / (1024.0 * 1024.0),
      allocator.free_list_size(), allocator.allocation_count(),
      allocator.release_count());
  LOG(info) << uv.pax_connection_info_.size() << " pax connection infos";
}

}  // namespace motis::paxmon
