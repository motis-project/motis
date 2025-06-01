#pragma once

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/point_rtree.h"

namespace motis {

std::vector<nigiri::location_idx_t> get_stops_with_traffic(
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    point_rtree<nigiri::location_idx_t> const&,
    osr::location const&,
    double const distance,
    nigiri::location_idx_t const not_equal_to =
        nigiri::location_idx_t::invalid());

}  // namespace motis