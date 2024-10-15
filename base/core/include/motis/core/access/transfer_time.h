#pragma once

#include <optional>

#include "motis/core/schedule/station.h"
#include "motis/core/schedule/time.h"

namespace motis {

std::optional<duration> get_transfer_time_between_platforms(
    station const& from_station, std::optional<uint16_t> from_platform,
    station const& to_station, std::optional<uint16_t> to_platform);

}  // namespace motis
