#pragma once

#include <optional>

#include "nigiri/types.h"
#include "motis/config.h"
#include "motis/fwd.h"

namespace motis {

void route_shapes(osr::ways const&,
                  osr::lookup const&,
                  osr::platforms const&,
                  nigiri::timetable&,
                  nigiri::shapes_storage&,
                  config::timetable::route_shapes const&,
                  std::array<bool, nigiri::kNumClasses> const&);

}  // namespace motis
