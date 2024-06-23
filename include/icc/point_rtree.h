#pragma once

#include <array>

#include "rtree.h"

#include "geo/latlng.h"

namespace icc {

template <typename T, typename Fn>
concept RtreePosHandler = requires(geo::latlng const& pos, T const x, Fn&& f) {
  { std::forward<Fn>(f)(pos, x) };
};

template <typename T>
struct point_rtree {
  point_rtree() : rtree_{rtree_new()} {}

  ~point_rtree() {
    if (rtree_ != nullptr) {
      rtree_free(rtree_);
    }
  }

  point_rtree(point_rtree const&) = delete;
  point_rtree(point_rtree&& o) {
    if (this != &o) {
      rtree_ = o.rtree_;
      o.rtree_ = nullptr;
    }
  }

  point_rtree& operator=(point_rtree const&) = delete;
  point_rtree& operator=(point_rtree&& o) {
    if (this != &o) {
      rtree_ = o.rtree_;
      o.rtree_ = nullptr;
    }
  }

  void add(geo::latlng const& pos, T const t) {
    auto const min_corner = std::array{pos.lng(), pos.lat()};
    rtree_insert(
        rtree_, min_corner.data(), nullptr,
        reinterpret_cast<void*>(static_cast<std::size_t>(cista::to_idx(t))));
  }

  template <typename Fn>
  void in_radius(geo::latlng const& x, double distance, Fn&& fn) const {
    find(x, [&](geo::latlng const& pos, T const item) {
      if (geo::distance(x, pos) < distance) {
        fn(item);
      }
    });
  }

  template <typename Fn>
  void find(geo::latlng const& x, Fn&& fn) const {
    find({x.lat() - 0.01, x.lng() - 0.01}, {x.lat() + 0.01, x.lng() + 0.01},
         std::forward<Fn>(fn));
  }

  template <typename Fn>
  void find(geo::latlng const& a, geo::latlng const& b, Fn&& fn) const {
    auto const min =
        std::array{std::min(a.lng_, b.lng_), std::min(a.lat_, b.lat_)};
    auto const max =
        std::array{std::max(a.lng_, b.lng_), std::max(a.lat_, b.lat_)};
    rtree_search(
        rtree_, min.data(), max.data(),
        [](double const* min, double const* /* max */, void const* item,
           void* udata) {
          if constexpr (RtreePosHandler<T, Fn>) {
            (*reinterpret_cast<Fn*>(udata))(
                geo::latlng{min[1], min[0]},
                T{static_cast<cista::base_t<T>>(
                    reinterpret_cast<std::size_t>(item))});
          } else {
            (*reinterpret_cast<Fn*>(udata))(T{static_cast<cista::base_t<T>>(
                reinterpret_cast<std::size_t>(item))});
          }
          return true;
        },
        &fn);
  }

  rtree* rtree_{nullptr};
};

}  // namespace icc