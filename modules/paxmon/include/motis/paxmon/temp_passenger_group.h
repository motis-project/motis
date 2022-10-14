#pragma once

#include <optional>
#include <vector>

#include "motis/core/schedule/time.h"

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/group_route.h"
#include "motis/paxmon/index_types.h"
#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct temp_group_route {
  std::optional<local_group_route_index> index_{};
  float probability_{};
  compact_journey journey_;
  motis::time planned_arrival_time_{INVALID_TIME};
  std::int16_t estimated_delay_{};
  route_source_flags source_flags_{route_source_flags::NONE};
  bool planned_{};
};

struct temp_passenger_group {
  passenger_group_index id_{};
  data_source source_{};
  std::uint16_t passengers_{1};
  std::vector<temp_group_route> routes_;
};

struct temp_passenger_group_with_route {
  passenger_group_index group_id_{};
  data_source source_{};
  std::uint16_t passengers_{1};
  temp_group_route route_;
};

}  // namespace motis::paxmon
