#pragma once

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

// Starts counting timetable places at the last OSM place.
using adr_extra_place_idx_t =
    cista::strong<std::uint32_t, struct adr_extra_place_idx_>;

using location_place_map_t =
    vector_map<nigiri::location_idx_t, adr_extra_place_idx_t>;

location_place_map_t adr_extend_tt(nigiri::timetable const&,
                                   adr::area_database const*,
                                   adr::typeahead&);

}  // namespace motis