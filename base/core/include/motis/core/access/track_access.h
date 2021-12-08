#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "motis/core/schedule/schedule.h"

namespace motis {

std::optional<uint16_t> get_track_index(schedule const&,
                                        std::string_view track_name);

}  // namespace motis
