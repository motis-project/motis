#pragma once

#include "nigiri/timetable.h"

#include "motis/point_rtree.h"

namespace motis {

inline point_rtree<nigiri::location_idx_t> create_location_rtree(
    nigiri::timetable const& tt) {
  auto t = point_rtree<nigiri::location_idx_t>{};
  for (auto i = nigiri::location_idx_t{0U}; i != tt.n_locations(); ++i) {
    t.add(tt.locations_.coordinates_[i], i);
  }
  return t;
}

}  // namespace motis