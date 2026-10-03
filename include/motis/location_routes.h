#pragma once

#include "nigiri/location_routes.h"
#include "nigiri/timetable.h"

namespace motis {

inline nigiri::hash_set<std::string_view> get_location_routes(
    nigiri::timetable const& tt, nigiri::location_idx_t const l) {
  auto names = nigiri::hash_set<std::string_view>{};
  nigiri::for_each_route_at_stop(tt, l, [&](nigiri::route_idx_t const r) {
    for (auto const t : tt.route_transport_ranges_[r]) {
      names.emplace(tt.transport_name(t));
    }
  });
  return names;
}

}  // namespace motis
