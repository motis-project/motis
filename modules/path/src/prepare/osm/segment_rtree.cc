#include "motis/path/prepare/osm/segment_rtree.h"

#include "boost/geometry/geometries/box.hpp"
#include "boost/geometry/geometries/point.hpp"
#include "boost/geometry/geometries/register/segment.hpp"
#include "boost/geometry/index/rtree.hpp"
#include "boost/iterator/function_output_iterator.hpp"

#include "geo/detail/register_box.h"
#include "geo/detail/register_latlng.h"

BOOST_GEOMETRY_REGISTER_SEGMENT(motis::path::segment_rtree::segment_t,
                                geo::latlng, first, second);

namespace bgi = boost::geometry::index;

namespace motis::path {

struct segment_rtree::impl {
  using rtree_t = bgi::rtree<value_t, bgi::quadratic<16>>;

  impl() = default;
  explicit impl(std::vector<value_t> const& index) : rtree_(index) {}

  std::vector<std::pair<double, size_t>> intersects_radius_with_distance(
      geo::latlng const& center, double const max_radius) const {
    std::vector<std::pair<double, size_t>> results;
    rtree_.query(bgi::intersects(geo::box{center, max_radius}),
                 boost::make_function_output_iterator([&](auto&& v) {
                   auto const dist =
                       boost::geometry::distance(v.first, center) *
                       geo::kEarthRadiusMeters;
                   if (dist >= max_radius) {
                     return;
                   }
                   results.emplace_back(dist, v.second);
                 }));

    std::sort(begin(results), end(results));
    return results;
  }

  rtree_t rtree_;
};

segment_rtree::segment_rtree()
    : impl_(std::make_unique<segment_rtree::impl>()) {}
segment_rtree::~segment_rtree() = default;

segment_rtree::segment_rtree(std::vector<value_t> const& index)
    : impl_(std::make_unique<segment_rtree::impl>(index)) {}

segment_rtree::segment_rtree(segment_rtree&&) noexcept = default;
segment_rtree& segment_rtree::operator=(segment_rtree&&) noexcept = default;

std::vector<std::pair<double, size_t>>
segment_rtree::intersects_radius_with_distance(geo::latlng const& center,
                                               double const max_radius) const {
  return impl_->intersects_radius_with_distance(center, max_radius);
}

}  // namespace motis::path
