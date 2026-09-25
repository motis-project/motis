#pragma once

#include "nigiri/timetable.h"

namespace motis {

// The routes stopping at a stop: its own and the ones at its virtual locations
// (transfers.txt rules move trips there, the stop is still where they stop).
template <typename Fn>
void for_each_route_at(nigiri::timetable const& tt,
                       nigiri::location_idx_t const l,
                       Fn&& fn) {
  for (auto const r : tt.location_routes_[l]) {
    fn(r);
  }
  for (auto const c : tt.locations_.children_[l]) {
    if (tt.locations_.types_[c] == nigiri::location_type::kVirt) {
      for (auto const r : tt.location_routes_[c]) {
        fn(r);
      }
    }
  }
}

inline bool has_routes(nigiri::timetable const& tt,
                       nigiri::location_idx_t const l) {
  auto any = false;
  for_each_route_at(tt, l, [&](nigiri::route_idx_t) { any = true; });
  return any;
}

inline nigiri::hash_set<std::string_view> get_location_routes(
    nigiri::timetable const& tt, nigiri::location_idx_t const l) {
  auto names = nigiri::hash_set<std::string_view>{};
  for_each_route_at(tt, l, [&](nigiri::route_idx_t const r) {
    for (auto const t : tt.route_transport_ranges_[r]) {
      names.emplace(tt.transport_name(t));
    }
  });
  return names;
}

}  // namespace motis