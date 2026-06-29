#include "motis/get_stops_with_traffic.h"

#include "osr/location.h"
#include "utl/helpers/algorithm.h"
#include "utl/pipes/for_each.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace n = nigiri;

namespace motis {

std::vector<n::location_idx_t> get_stops_with_traffic(
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    point_rtree<n::location_idx_t> const& rtree,
    osr::location const& pos,
    double const distance,
    n::location_idx_t const not_equal_to) {
  auto ret = std::vector<n::location_idx_t>{};
  rtree.in_radius(pos.pos_, distance, [&](n::location_idx_t const l) {
    if (tt.location_routes_[l].empty() &&
        (rtt == nullptr || rtt->location_rt_transports_[l].empty())) {
      return;
    }
    if (l == not_equal_to) {
      return;
    }
    ret.emplace_back(l);
  });
  return ret;
}

std::vector<nigiri::location_idx_t> get_stops_with_unique_routes(
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    point_rtree<n::location_idx_t> const& rtree,
    osr::location const& pos,
    double const distance) {
  auto ret = std::vector<n::location_idx_t>{};
  auto unique_routes = hash_set<nigiri::route_idx_t>{};
  rtree.in_radius(pos.pos_, distance, [&](n::location_idx_t const l) {
    if (tt.location_routes_[l].empty() &&
        (rtt == nullptr || rtt->location_rt_transports_[l].empty())) {
      return;
    }
    if (!utl::any_of(tt.location_routes_[l], [&](auto const& route) {
          return !unique_routes.contains(route);
        })) {
      return;
    }
    ret.emplace_back(l);
    for (auto const& route : tt.location_routes_[l]) {
      unique_routes.insert(route);
    }
  });
  return ret;
}

}  // namespace motis