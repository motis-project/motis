#pragma once

#include "date/tz.h"

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

// Starts counting timetable places at the last OSM place.
using adr_extra_place_idx_t =
    cista::strong<std::uint32_t, struct adr_extra_place_idx_>;

using tz_map_t = vector_map<adr_extra_place_idx_t, date::time_zone const*>;

struct adr_ext {
  vector_map<nigiri::location_idx_t, adr_extra_place_idx_t> location_place_;
  vector_map<adr_extra_place_idx_t, nigiri::routing::clasz_mask_t> place_clasz_;
  vector_map<adr_extra_place_idx_t, float> place_importance_;
};

date::time_zone const* get_tz(nigiri::timetable const&,
                              adr_ext const*,
                              tz_map_t const*,
                              nigiri::location_idx_t);

adr_ext adr_extend_tt(nigiri::timetable const&,
                      adr::area_database const*,
                      adr::typeahead&);

}  // namespace motis