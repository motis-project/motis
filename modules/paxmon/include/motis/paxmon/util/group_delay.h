#pragma once

#include <cstdint>

#include "motis/paxmon/universe.h"

namespace motis::paxmon::util {

struct group_delay_info {
  float estimated_delay_{};
  bool possibly_unreachable_{};
};

inline group_delay_info get_current_estimated_delay(
    universe const& uv, passenger_group_index const pgi,
    std::int16_t const unreachable_delay = 24 * 60) {
  auto info = group_delay_info{};
  auto at_least_one_active_route = false;
  for (auto const& gr : uv.passenger_groups_.routes(pgi)) {
    if (gr.probability_ != 0) {
      auto const route_est_delay =
          gr.destination_unreachable_ ? unreachable_delay : gr.estimated_delay_;
      info.estimated_delay_ += gr.probability_ * route_est_delay;
      at_least_one_active_route = true;
      if (gr.destination_unreachable_) {
        info.possibly_unreachable_ = true;
      }
    }
  }
  if (!at_least_one_active_route) {
    info.estimated_delay_ = unreachable_delay;
    info.possibly_unreachable_ = true;
  }
  return info;
}

inline std::int16_t get_scheduled_delay(universe const& uv,
                                        passenger_group_with_route const pgwr,
                                        std::int16_t const invalid_delay = 24 *
                                                                           60) {
  // assumption in this function: route 0 is the (only) planned route
  if (pgwr.route_ == 0) {
    return 0;
  }
  auto const this_planned_arrival =
      uv.passenger_groups_.route(pgwr).planned_arrival_time_;
  if (this_planned_arrival == INVALID_TIME) {
    return invalid_delay;
  }
  auto const original_planned_arrival =
      uv.passenger_groups_.route(passenger_group_with_route{pgwr.pg_, 0})
          .planned_arrival_time_;
  return static_cast<std::int16_t>(static_cast<int>(this_planned_arrival) -
                                   static_cast<int>(original_planned_arrival));
}

}  // namespace motis::paxmon::util
