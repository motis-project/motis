#pragma once

#include <cstdint>

#include "motis/core/schedule/schedule.h"

namespace motis::rt {

struct expanded_trip_index {
  std::uint32_t route_index_{};
  std::uint32_t index_in_route_{};
};

inline expanded_trip_index add_trip_to_new_expanded_route(
    schedule& sched, trip* trp, std::uint32_t new_route_id) {
  auto new_exp_route = sched.expanded_trips_.emplace_back();
  new_exp_route.emplace_back(trp);
  sched.route_to_expanded_routes_[new_route_id].emplace_back(
      new_exp_route.index());
  return {new_exp_route.index(), 0};
}

}  // namespace motis::rt
