#pragma once

#include <memory>
#include <vector>

#include "geo/box.h"
#include "geo/latlng.h"

namespace motis::path {

struct segment_rtree {
  using segment_t = std::pair<geo::latlng, geo::latlng>;
  using value_t = std::pair<segment_t, size_t>;

  segment_rtree();
  ~segment_rtree();

  explicit segment_rtree(std::vector<value_t> const&);

  segment_rtree(segment_rtree const&) noexcept = delete;
  segment_rtree& operator=(segment_rtree const&) noexcept = delete;
  segment_rtree(segment_rtree&&) noexcept;
  segment_rtree& operator=(segment_rtree&&) noexcept;

  std::vector<std::pair<double, size_t>> intersects_radius_with_distance(
      geo::latlng const& center, double max_radius) const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

template <typename C, typename F>
segment_rtree make_segment_rtree(C const& container, F fun) {
  auto i = 0UL;
  std::vector<segment_rtree::value_t> index;
  index.reserve(container.size());
  for (auto const& e : container) {
    index.emplace_back(fun(e), i++);
  }
  return segment_rtree{index};
}

template <typename C>
segment_rtree make_segment_rtree(C const& container) {
  auto i = 0UL;
  std::vector<segment_rtree::value_t> index;
  index.reserve(container.size());
  for (auto const& e : container) {
    index.emplace_back(e, i++);
  }
  return segment_rtree{index};
}

}  // namespace motis::path
