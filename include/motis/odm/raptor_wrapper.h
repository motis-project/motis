#pragma once

#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace motis::odm {

nigiri::routing::routing_result<nigiri::routing::raptor_stats> raptor_wrapper(
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    nigiri::routing::query,
    nigiri::direction,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace motis::odm