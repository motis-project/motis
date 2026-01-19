#pragma once

#include <optional>

#include "motis/config.h"
#include "motis/fwd.h"

namespace motis {

void compute_shapes(
    osr::ways const&,
    osr::lookup const&,
    osr::platforms const&,
    nigiri::timetable&,
    nigiri::shapes_storage&,
    std::optional<config::timetable::shapes_debug> const& = std::nullopt);

}  // namespace motis
