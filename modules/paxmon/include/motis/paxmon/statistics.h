#pragma once

#include <cstdint>
#include <vector>

namespace motis::paxmon {

struct system_statistics {
  std::uint64_t total_broken_interchanges_{};
  std::uint64_t total_affected_passengers_{};
  std::uint64_t delay_updates_{};
  std::uint64_t reroute_updates_{};
  std::uint64_t groups_ok_count_{};
  std::uint64_t groups_broken_count_{};

  std::uint64_t update_event_times_trip_edges_found_{};
  std::uint64_t update_event_times_dep_updated_{};
  std::uint64_t update_event_times_arr_updated_{};
  std::uint64_t total_updated_interchange_edges_{};

  std::uint64_t update_trip_route_count_{};
  std::uint64_t update_trip_route_trip_edges_found_{};
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
  std::uint64_t affected_groups_{};
  std::uint64_t affected_passengers_{};
  std::uint64_t ok_groups_{};
  std::uint64_t broken_groups_{};
  std::uint64_t broken_passengers_{};

  // totals
  std::uint64_t total_ok_groups_{};
  std::uint64_t total_broken_groups_{};

  // tracking
  std::uint64_t tracked_ok_groups_{};
  std::uint64_t tracked_broken_groups_{};
};

}  // namespace motis::paxmon
