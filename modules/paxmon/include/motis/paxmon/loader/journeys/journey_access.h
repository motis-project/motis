#pragma once

#include <functional>
#include <optional>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon {

struct journey_trip_segment {
  journey::trip const* trip_{};
  std::size_t from_{};
  std::size_t to_{};
};

std::vector<journey_trip_segment> get_journey_trip_segments(
    journey const& j, std::size_t enter_stop_idx, std::size_t exit_stop_idx);

void for_each_trip(
    journey const& j, schedule const& sched,
    std::function<void(trip const*, journey::stop const&, journey::stop const&,
                       std::optional<transfer_info> const&)> const& cb);

}  // namespace motis::paxmon
