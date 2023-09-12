#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "geo/point_rtree.h"

#include "motis/transfers/platform/platform.h"

namespace motis::transfers {

struct platform_index {

  explicit platform_index(platforms& pfs) : platforms_(std::move(pfs)) {
    make_point_rtree();
  }

  // Returns the number of platforms stored in the index.
  std::size_t size() const { return platforms_.size(); }

  // Returns the `i`-th platform stored in the index. `i` in [0, size() - 1].
  platform get_platform(std::size_t const i) const { return platforms_[i]; }

  // Returns a list of (distance, platform) tuples of platforms within a radius
  // around the given coordinate.
  std::vector<std::pair<double, platform>>
  get_platforms_in_radius_with_distance_info(geo::latlng const&,
                                             double const) const;

  // Returns a list of the indexes of stored platforms in the index within a
  // radius around the given platform. The given platform will not be included
  // in the output.
  std::vector<size_t> get_other_platforms_in_radius(platform const&,
                                                    double const) const;

private:
  // Generates a rtree using the stored platforms in the index.
  void make_point_rtree();

  // Returns a list of platforms within a radius around the given coordinate.
  [[maybe_unused]] platforms get_platforms_in_radius(geo::latlng const&,
                                                     double const) const;

  platforms platforms_;
  geo::point_rtree platform_index_;
};

}  // namespace motis::transfers