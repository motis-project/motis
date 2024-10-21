#pragma once

#include "adr/adr.h"
#include "adr/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis {

api::geocode_response suggestions_to_response(
    adr::typeahead const&,
    nigiri::timetable const*,
    tag_lookup const*,
    osr::ways const* w,
    osr::platforms const* pl,
    platform_matches_t const* matches,
    std::basic_string<adr::language_idx_t> const& lang_indices,
    std::vector<adr::token> const& token_pos,
    std::vector<adr::suggestion> const&);

}  // namespace motis