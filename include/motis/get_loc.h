#pragma once

#include "osr/platforms.h"
#include "osr/routing/route.h"

#include "nigiri/timetable.h"

#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

inline osr::location get_loc(
    nigiri::timetable const& tt,
    osr::ways const& w,
    osr::platforms const& pl,
    vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches,
    nigiri::location_idx_t const l) {
  auto const m = get_match(tt, matches, l);
  auto const lvl = m == osr::platform_idx_t::invalid() ? osr::level_t{0.F}
                                                       : pl.get_level(w, m);
  return {tt.locations_.coordinates_[l], lvl};
}

}  // namespace motis