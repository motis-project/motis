#pragma once

#include <cstdint>

namespace motis::paxforecast {

struct tick_statistics {
  tick_statistics& operator+=(tick_statistics const& rhs) {
    monitoring_events_ += rhs.monitoring_events_;
    group_routes_ += rhs.group_routes_;
    combined_groups_ += rhs.combined_groups_;
    major_delay_group_routes_ += rhs.major_delay_group_routes_;

    routing_requests_ += rhs.routing_requests_;
    alternatives_found_ += rhs.alternatives_found_;

    rerouted_group_routes_ += rhs.rerouted_group_routes_;
    removed_group_routes_ += rhs.removed_group_routes_;
    major_delay_group_routes_with_alternatives_ +=
        rhs.major_delay_group_routes_with_alternatives_;

    t_find_alternatives_ += rhs.t_find_alternatives_;
    t_add_alternatives_ += rhs.t_add_alternatives_;
    t_passenger_behavior_ += rhs.t_passenger_behavior_;
    t_calc_load_forecast_ += rhs.t_calc_load_forecast_;
    t_load_forecast_fbs_ += rhs.t_load_forecast_fbs_;
    t_write_load_forecast_ += rhs.t_write_load_forecast_;
    t_publish_load_forecast_ += rhs.t_publish_load_forecast_;
    t_total_load_forecast_ += rhs.t_total_load_forecast_;
    t_update_tracked_groups_ += rhs.t_update_tracked_groups_;
    t_total_ += rhs.t_total_;

    return *this;
  }

  friend tick_statistics operator+(tick_statistics lhs,
                                   tick_statistics const& rhs) {
    lhs += rhs;
    return lhs;
  }

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
