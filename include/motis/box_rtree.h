#pragma once

#include <algorithm>
#include <array>

#include "cista/strong.h"

#include "rtree.h"

#include "geo/box.h"
#include "geo/latlng.h"

namespace motis {

template <typename T, typename Fn>
concept BoxRtreePosHandler = requires(geo::box const& b, T const x, Fn&& f) {
  { std::forward<Fn>(f)(b, x) };
};

template <typename T>
struct box_rtree {
  box_rtree() : rtree_{rtree_new()} {}

  ~box_rtree() {
    if (rtree_ != nullptr) {
      rtree_free(rtree_);
    }
  }

  box_rtree(box_rtree const& o) {
    if (this != &o) {
      if (rtree_ != nullptr) {
        rtree_free(rtree_);
      }
      rtree_ = rtree_clone(o.rtree_);
    }
  }

  box_rtree(box_rtree&& o) {
    if (this != &o) {
      rtree_ = o.rtree_;
      o.rtree_ = nullptr;
    }
  }

  box_rtree& operator=(box_rtree const& o) {
    if (this != &o) {
      if (rtree_ != nullptr) {
        rtree_free(rtree_);
      }
      rtree_ = rtree_clone(o.rtree_);
    }
    return *this;
  }

  box_rtree& operator=(box_rtree&& o) {
    if (this != &o) {
      rtree_ = o.rtree_;
      o.rtree_ = nullptr;
    }
    return *this;
  }

  void add(geo::box const& b, T const t) {
    auto const min_corner = b.min_.lnglat();
    auto const max_corner = b.max_.lnglat();
    rtree_insert(
        rtree_, min_corner.data(), max_corner.data(),
        reinterpret_cast<void*>(static_cast<std::size_t>(cista::to_idx(t))));
  }

  void remove(geo::box const& b, T const t) {
    auto const min_corner = b.min_.lnglat();
    auto const max_corner = b.max_.lnglat();
    rtree_delete(
        rtree_, min_corner.data(), max_corner.data(),
        reinterpret_cast<void*>(static_cast<std::size_t>(cista::to_idx(t))));
  }

  std::vector<T> in_radius(geo::latlng const& x, double distance) const {
    auto ret = std::vector<T>{};
    in_radius(x, distance, [&](auto&& item) { ret.emplace_back(item); });
    return ret;
  }

  template <typename Fn>
  void in_radius(geo::latlng const& x, double distance, Fn&& fn) const {
    auto const rad_sq = distance * distance;
    auto const approx_distance_lng_degrees =
        geo::approx_distance_lng_degrees(x);
    find(geo::box{x, distance}, [&](geo::box const& box, T const item) {
      auto const closest =
          geo::latlng{std::clamp(x.lat(), box.min_.lat(), box.max_.lat()),
                      std::clamp(x.lng(), box.min_.lng(), box.max_.lng())};
      if (geo::approx_squared_distance(x, closest,
                                       approx_distance_lng_degrees) < rad_sq) {
        fn(item);
      }
    });
  }

  template <typename Fn>
  void find(geo::box const& b, Fn&& fn) const {
    auto const min = b.min_.lnglat();
    auto const max = b.max_.lnglat();
    rtree_search(
        rtree_, min.data(), max.data(),
        [](double const* min_corner, double const* max_corner, void const* item,
           void* udata) {
          if constexpr (BoxRtreePosHandler<T, Fn>) {
            (*reinterpret_cast<Fn*>(udata))(
                geo::box{geo::latlng{min_corner[1], min_corner[0]},
                         geo::latlng{max_corner[1], max_corner[0]}},
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

  template <typename Fn>
  void find(geo::latlng const& pos, Fn&& fn) const {
    return find(geo::box{pos, pos}, std::forward<Fn>(fn));
  }

  rtree* rtree_{nullptr};
};

}  // namespace motis
