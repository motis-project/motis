#include "motis/route_coverage.h"

#include <algorithm>

#include "utl/helpers/algorithm.h"

#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis {

constexpr auto const kMaxStopsPerCoveredRoute = std::uint8_t{2U};

bool route_coverage::select(n::timetable const& tt,
                            n::location_idx_t const origin,
                            unsigned const min_routes) {
  selected_.clear();
  stops_per_route_.clear();
  if (origin != n::location_idx_t::invalid()) {
    for (auto const r : tt.location_routes_[origin]) {
      stops_per_route_[r] = 1U;
    }
  }

  auto const needs_more_stops = [&](n::route_idx_t const r) {
    auto const it = stops_per_route_.find(r);
    return it == end(stops_per_route_) || it->second < kMaxStopsPerCoveredRoute;
  };

  utl::sort(reached_);
  for (auto const& [_, l] : reached_) {
    if (stops_per_route_.size() >= min_routes) {
      break;
    }
    if (utl::none_of(tt.location_routes_[l], needs_more_stops)) {
      continue;
    }
    selected_.push_back(l);
    for (auto const r : tt.location_routes_[l]) {
      auto& n_stops = stops_per_route_[r];
      n_stops = std::min(static_cast<std::uint8_t>(n_stops + 1U),
                         kMaxStopsPerCoveredRoute);
    }
  }
  utl::sort(selected_);

  return stops_per_route_.size() >= min_routes;
}

bool route_coverage::is_selected(n::location_idx_t const l) const {
  return std::ranges::binary_search(selected_, l);
}

}  // namespace motis
