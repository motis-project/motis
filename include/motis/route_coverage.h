#pragma once

#include <utility>
#include <vector>

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/types.h"

namespace motis {

struct route_coverage {
  // Picks stops from reached_ in ascending duration until min_routes routes
  // are served, taking at most two stops per route. Routes of origin (if
  // valid) count as served by one stop. Returns whether min_routes is met.
  bool select(nigiri::timetable const&,
              nigiri::location_idx_t origin,
              unsigned min_routes);

  bool is_selected(nigiri::location_idx_t) const;

  void clear() {
    reached_.clear();
    selected_.clear();
    stops_per_route_.clear();
  }

  std::vector<std::pair<nigiri::duration_t, nigiri::location_idx_t>> reached_;
  std::vector<nigiri::location_idx_t> selected_;
  hash_map<nigiri::route_idx_t, std::uint8_t> stops_per_route_;
};

}  // namespace motis
