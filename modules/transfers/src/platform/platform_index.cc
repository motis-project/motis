#include "motis/transfers/platform/platform_index.h"

#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

namespace motis::transfers {

void platform_index::make_point_rtree() {
  platform_index_ =
      geo::make_point_rtree(platforms_, [](auto const& pf) { return pf.loc_; });
}

std::vector<std::pair<double, platform>>
platform_index::get_platforms_in_radius_with_distance_info(
    geo::latlng const& coord, double const radius) const {
  return utl::all(platform_index_.in_radius_with_distance(coord, radius)) |
         utl::transform([this](std::pair<double, std::size_t> res) {
           return std::pair<double, platform>(res.first,
                                              get_platform(res.second));
         }) |
         utl::vec();
}

std::vector<std::size_t> platform_index::get_other_platforms_in_radius(
    platform const& pf, double const radius) const {
  return utl::all(platform_index_.in_radius(pf.loc_, radius)) |
         utl::remove_if(
             [this, &pf](std::size_t i) { return get_platform(i) == pf; }) |
         utl::vec();
}

}  // namespace motis::transfers
