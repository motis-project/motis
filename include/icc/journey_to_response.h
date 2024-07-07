#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "icc-api/icc-api.h"
#include "icc/types.h"

namespace icc {

using place_t = std::variant<osr::location, nigiri::location_idx_t>;

inline std::ostream& operator<<(std::ostream& out, place_t const p) {
  return std::visit([&](auto const l) -> std::ostream& { return out << l; }, p);
}

api::Itinerary journey_to_response(
    osr::ways const&,
    osr::lookup const&,
    nigiri::timetable const&,
    osr::platforms const& pl,
    nigiri::rt_timetable const*,
    osr::bitvec<osr::node_idx_t> const* blocked,
    vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches,
    bool const wheelchair,
    nigiri::routing::journey const&,
    place_t const& start,
    place_t const& dest);

}  // namespace icc