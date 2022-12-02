#pragma once

#include <cstdint>

namespace motis::paxforecast {

struct tick_statistics {
  std::uint64_t system_time_{};

  std::uint64_t monitoring_events_{};
  std::uint64_t group_routes_{};
  std::uint64_t combined_groups_{};
  std::uint64_t major_delay_group_routes_{};

  std::uint64_t routing_requests_{};
  std::uint64_t alternatives_found_{};

  std::uint64_t rerouted_group_routes_{};
  std::uint64_t removed_group_routes_{};
  std::uint64_t major_delay_group_routes_with_alternatives_{};

  // timing (ms)
  std::uint64_t t_find_alternatives_{};
  std::uint64_t t_add_alternatives_{};
  std::uint64_t t_passenger_behavior_{};
  std::uint64_t t_calc_load_forecast_{};
  std::uint64_t t_load_forecast_fbs_{};
  std::uint64_t t_write_load_forecast_{};
  std::uint64_t t_publish_load_forecast_{};
  std::uint64_t t_total_load_forecast_{};
  std::uint64_t t_update_tracked_groups_{};
  std::uint64_t t_total_{};
};

}  // namespace motis::paxforecast
