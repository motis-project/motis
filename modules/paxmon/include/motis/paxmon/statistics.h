#pragma once

#include <cstdint>
#include <vector>

namespace motis {
struct schedule;
}  // namespace motis

namespace motis::paxmon {

struct system_statistics {
  std::uint64_t total_broken_interchanges_{};
  std::uint64_t delay_updates_{};
  std::uint64_t reroute_updates_{};

  std::uint64_t update_event_times_trip_edges_found_{};
  std::uint64_t update_event_times_dep_updated_{};
  std::uint64_t update_event_times_arr_updated_{};
  std::uint64_t total_updated_interchange_edges_{};

  std::uint64_t update_trip_route_count_{};
  std::uint64_t update_trip_route_trip_edges_found_{};

  std::uint64_t update_track_count_{};
  std::uint64_t update_track_trip_found_{};
};

struct tick_statistics {
  std::uint64_t system_time_{};

  // rt update counts
  std::uint64_t rt_updates_{};
  std::uint64_t rt_delay_updates_{};
  std::uint64_t rt_reroute_updates_{};
  std::uint64_t rt_track_updates_{};
  std::uint64_t rt_free_text_updates_{};

  // rt delay details
  std::uint64_t rt_delay_event_updates_{};
  std::uint64_t rt_delay_is_updates_{};
  std::uint64_t rt_delay_propagation_updates_{};
  std::uint64_t rt_delay_forecast_updates_{};
  std::uint64_t rt_delay_repair_updates_{};
  std::uint64_t rt_delay_schedule_updates_{};

  // affected passengers
  std::uint64_t affected_group_routes_{};
  std::uint64_t ok_group_routes_{};
  std::uint64_t broken_group_routes_{};
  std::uint64_t major_delay_group_routes_{};

  // timing (ms)
  std::uint64_t t_reachability_{};
  std::uint64_t t_localization_{};
  std::uint64_t t_update_load_{};
  std::uint64_t t_fbs_events_{};
  std::uint64_t t_publish_{};
  std::uint64_t t_rt_updates_applied_total_{};
};

struct graph_statistics {
  std::uint64_t passenger_groups_{};
  std::uint64_t passengers_{};
  std::uint64_t nodes_{};
  std::uint64_t canceled_nodes_{};
  std::uint64_t edges_{};
  std::uint64_t trip_edges_{};
  std::uint64_t interchange_edges_{};
  std::uint64_t wait_edges_{};
  std::uint64_t through_edges_{};
  std::uint64_t disabled_edges_{};
  std::uint64_t canceled_edges_{};
  std::uint64_t broken_edges_{};
  std::uint64_t trips_{};
  std::uint64_t stations_{};
  std::uint64_t edges_over_capacity_{};
  std::uint64_t trips_over_capacity_{};
  std::uint64_t passenger_group_routes_{};
  std::uint64_t broken_passenger_group_routes_{};
};

struct universe;

graph_statistics calc_graph_statistics(schedule const& sched,
                                       universe const& uv);

}  // namespace motis::paxmon
