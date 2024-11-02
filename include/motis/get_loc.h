#pragma once

#include "osr/platforms.h"
#include "osr/routing/route.h"

#include "nigiri/timetable.h"

#include "motis/constants.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

inline osr::location get_loc(
    nigiri::timetable const& tt,
    osr::ways const& w,
    osr::platforms const& pl,
    vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches,
    nigiri::location_idx_t const l) {
  auto pos = tt.locations_.coordinates_[l];
  if (matches[l] != osr::platform_idx_t::invalid()) {
    auto const center = get_platform_center(pl, w, matches[l]);
    if (center.has_value() && geo::distance(*center, pos) < kMaxAdjust) {
      pos = *center;
    }
  }
  auto const lvl = matches[l] == osr::platform_idx_t::invalid()
                       ? osr::level_t{0.F}
                       : pl.get_level(w, matches[l]);
  return {pos, lvl};
}

}  // namespace motis