#pragma once

#include "adr/adr.h"
#include "adr/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"

namespace motis {

api::geocode_response suggestions_to_response(
    adr::typeahead const&,
    nigiri::timetable const&,
    std::basic_string<adr::language_idx_t> const& lang_indices,
    std::vector<adr::token> const& token_pos,
    std::vector<adr::suggestion> const&);

}  // namespace motis