#include "motis/tripbased/lower_bounds.h"

#include "utl/to_vec.h"

#include "motis/hash_map.h"

namespace motis::tripbased {

lower_bounds calc_lower_bounds(schedule const& sched,
                               trip_based_query const& query) {
  mcd::hash_map<unsigned, std::vector<simple_edge>> travel_time_lb_graph_edges;
  for (auto const& e : query.start_edges_) {
    auto const from_station = query.start_station_;
    auto const to_station = e.station_id_;
    travel_time_lb_graph_edges[to_station].emplace_back(
        simple_edge{from_station, e.duration_});
  }

  for (auto const& e : query.destination_edges_) {
    auto const from_station = e.station_id_;
    auto const to_station = query.destination_station_;
    travel_time_lb_graph_edges[to_station].emplace_back(
        simple_edge{from_station, e.duration_});
  }

  lower_bounds lbs(
      query.dir_ == search_dir::FWD ? sched.travel_time_lower_bounds_fwd_
                                    : sched.travel_time_lower_bounds_bwd_,
      utl::to_vec(query.meta_destinations_,
                  [](station_id station) { return static_cast<int>(station); }),
      travel_time_lb_graph_edges);
  lbs.travel_time_.run();
  return lbs;
}

}  // namespace motis::tripbased
