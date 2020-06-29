#pragma once

#include <functional>
#include <optional>

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon {

journey::trip const* get_journey_trip(journey const& j,
                                      std::size_t enter_stop_idx);

void for_each_trip(
    journey const& j, schedule const& sched,
    std::function<void(extern_trip const&, journey::stop const&,
                       journey::stop const&,
                       std::optional<transfer_info> const&)> const& cb);

}  // namespace motis::paxmon
