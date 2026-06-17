#pragma once

#include <vector>

#include "utl/concat.h"

#include "adr/typeahead.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/adr_extend_tt.h"

namespace motis {

inline void add_with_children(nigiri::timetable const& tt,
                              std::vector<nigiri::location_idx_t>& locations,
                              nigiri::location_idx_t const x) {
  locations.emplace_back(x);
  utl::concat(locations, tt.locations_.children_[x]);
  for (auto const& c : tt.locations_.children_[x]) {
    utl::concat(locations, tt.locations_.children_[c]);
  }
}

inline void add_location(nigiri::timetable const& tt,
                         adr::typeahead const* t,
                         adr_ext const* ae,
                         std::vector<nigiri::location_idx_t>& locations,
                         nigiri::location_idx_t const l,
                         bool const exact = false) {
  if (exact) {
    locations.emplace_back(l);
    return;
  }

  add_with_children(tt, locations, l);

  if (t != nullptr && ae != nullptr &&
      ae->location_place_[l] != adr_extra_place_idx_t::invalid()) {
    auto const place_idx =
        adr::place_idx_t{t->ext_start_ + cista::to_idx(ae->location_place_[l])};
    for (auto const id : t->place_osm_ids_[place_idx]) {
      auto const eq = nigiri::location_idx_t{
          static_cast<cista::base_t<nigiri::location_idx_t>>(id)};
      if (eq == l) {
        continue;
      }
      add_with_children(tt, locations, eq);
    }
    return;
  }

  auto const l_name = tt.get_default_translation(tt.locations_.names_[l]);
  for (auto const eq : tt.locations_.equivalences_[l]) {
    if (tt.get_default_translation(tt.locations_.names_[eq]) == l_name) {
      add_with_children(tt, locations, eq);
    }
  }
}

}  // namespace motis
