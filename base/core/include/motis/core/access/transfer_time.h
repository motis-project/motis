#pragma once

#include <cstdint>
#include <optional>

#include "motis/core/schedule/station.h"

namespace motis {

std::optional<int32_t> get_transfer_time_between_platforms(
    station const& from_station, std::optional<uint16_t> from_platform,
    station const& to_station, std::optional<uint16_t> to_platform);

}  // namespace motis
