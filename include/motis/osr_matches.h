#pragma once

#include <memory>

#include "osr/lookup.h"

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/types.h"

namespace motis {

using osr_matches_t = vector_map<nigiri::location_idx_t, osr::match_t>;

osr_matches_t get_osr_matches(nigiri::timetable const&,
                              osr::ways const&,
                              osr::lookup const&,
                              osr::platforms const&,
                              platform_matches_t const&);

}  // namespace motis