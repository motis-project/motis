#pragma once

#include "nigiri/timetable.h"

namespace motis {

inline nigiri::hash_set<std::string_view> get_location_routes(
    nigiri::timetable const& tt, nigiri::location_idx_t const l) {
  auto names = nigiri::hash_set<std::string_view>{};
  for (auto const r : tt.location_routes_[l]) {
    for (auto const t : tt.route_transport_ranges_[r]) {
      names.emplace(tt.transport_name(t));
    }
  }
  return names;
}

}  // namespace motis