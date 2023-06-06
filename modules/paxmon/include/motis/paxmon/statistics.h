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
  tick_statistics& operator+=(tick_statistics const& rhs) {
    rt_updates_ += rhs.rt_updates_;
    rt_delay_updates_ += rhs.rt_delay_updates_;
    rt_reroute_updates_ += rhs.rt_reroute_updates_;
    rt_track_updates_ += rhs.rt_track_updates_;
    rt_free_text_updates_ += rhs.rt_free_text_updates_;
    rt_trip_formation_updates_ += rhs.rt_trip_formation_updates_;

    rt_delay_event_updates_ += rhs.rt_delay_event_updates_;
    rt_delay_is_updates_ += rhs.rt_delay_is_updates_;
    rt_delay_propagation_updates_ += rhs.rt_delay_propagation_updates_;
    rt_delay_forecast_updates_ += rhs.rt_delay_forecast_updates_;
    rt_delay_repair_updates_ += rhs.rt_delay_repair_updates_;
    rt_delay_schedule_updates_ += rhs.rt_delay_schedule_updates_;

    affected_group_routes_ += rhs.affected_group_routes_;
    ok_group_routes_ += rhs.ok_group_routes_;
    broken_group_routes_ += rhs.broken_group_routes_;
    major_delay_group_routes_ += rhs.major_delay_group_routes_;

    t_reachability_ += rhs.t_reachability_;
    t_localization_ += rhs.t_localization_;
    t_update_load_ += rhs.t_update_load_;
    t_fbs_events_ += rhs.t_fbs_events_;
    t_publish_ += rhs.t_publish_;
    t_rt_updates_applied_total_ += rhs.t_rt_updates_applied_total_;

    return *this;
  }

  friend tick_statistics operator+(tick_statistics lhs,
                                   tick_statistics const& rhs) {
    lhs += rhs;
    return lhs;
  }

  std::uint64_t system_time_{};

  // rt update counts
  std::uint64_t rt_updates_{};
  std::uint64_t rt_delay_updates_{};
  std::uint64_t rt_reroute_updates_{};
  std::uint64_t rt_track_updates_{};
  std::uint64_t rt_free_text_updates_{};
  std::uint64_t rt_trip_formation_updates_{};

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
