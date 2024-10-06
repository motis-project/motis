#pragma once

#include <array>

#include "cista/strong.h"

#include "rtree.h"

#include "geo/box.h"
#include "geo/latlng.h"

namespace motis {

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
    return *this;
  }

  void add(geo::latlng const& pos, T const t) {
    auto const min_corner = std::array{pos.lng(), pos.lat()};
    rtree_insert(
        rtree_, min_corner.data(), nullptr,
        reinterpret_cast<void*>(static_cast<std::size_t>(cista::to_idx(t))));
  }

  std::vector<T> in_radius(geo::latlng const& x, double distance) const {
    auto ret = std::vector<T>{};
    in_radius(x, distance, [&](auto&& item) { ret.emplace_back(item); });
    return ret;
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
    find({{x.lat() - 0.01, x.lng() - 0.01}, {x.lat() + 0.01, x.lng() + 0.01}},
         std::forward<Fn>(fn));
  }

  template <typename Fn>
  void find(geo::box const& b, Fn&& fn) const {
    auto const min = b.min_.lnglat();
    auto const max = b.max_.lnglat();
    rtree_search(
        rtree_, min.data(), max.data(),
        [](double const* pos, double const* /* max */, void const* item,
           void* udata) {
          if constexpr (RtreePosHandler<T, Fn>) {
            (*reinterpret_cast<Fn*>(udata))(
                geo::latlng{pos[1], pos[0]},
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

}  // namespace motis