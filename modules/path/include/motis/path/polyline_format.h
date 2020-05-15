#pragma once

#include <cmath>

#include <iostream>

#include "geo/polyline.h"

namespace motis::path {

constexpr int64_t kPolylineMaskBits = 5;
constexpr int64_t kPolylineCurrMask = 0b11111;
constexpr int64_t kPolylineRestMask = ~kPolylineCurrMask;

// std::pow is sadly not constexpr
inline constexpr int64_t pow(int64_t const base, int64_t const exp) {
  return exp == 0 ? 1 : base * pow(base, exp - 1);
}

template <int64_t Precision = 5>
struct polyline_encoder {
  static constexpr auto const kPrecision = pow(10, Precision);

  void reset() {
    last_lat_ = 0;
    last_lng_ = 0;
    buf_.clear();
  }

  void push(geo::latlng const ll) {
    int64_t const lat = std::llround(ll.lat_ * kPrecision);
    int64_t const lng = std::llround(ll.lng_ * kPrecision);

    push_difference(lat - last_lat_);
    push_difference(lng - last_lng_);

    last_lat_ = lat;
    last_lng_ = lng;
  }

  bool push_nonzero_diff(geo::latlng const ll) {
    int64_t const lat = std::llround(ll.lat_ * kPrecision);
    int64_t const lng = std::llround(ll.lng_ * kPrecision);

    auto const lat_diff = lat - last_lat_;
    auto const lng_diff = lng - last_lng_;
    if (lat_diff != 0 || lng_diff != 0) {
      push_difference(lat_diff);
      push_difference(lng_diff);
      last_lat_ = lat;
      last_lng_ = lng;
      return true;
    } else {
      return false;
    }
  }

  void push_difference(int64_t diff) {
    auto tmp = diff << 1;
    if (diff < 0) {
      tmp = ~tmp;
    }

    for (auto i = 0; i < sizeof(int64_t) * 8; i += kPolylineMaskBits) {
      auto curr = tmp & kPolylineCurrMask;
      auto rest = tmp & kPolylineRestMask;

      if (rest != 0) {
        curr |= 0x20;
      }

      buf_.push_back(static_cast<char>(curr + 63));
      tmp = tmp >> kPolylineMaskBits;

      if (rest == 0) {
        break;
      }
    }
  }

  int64_t last_lat_{0};
  int64_t last_lng_{0};
  std::string buf_;
};

template <int64_t Precision = 5>
std::string encode_polyline(geo::polyline const& polyline) {
  polyline_encoder<Precision> enc;
  for (auto const ll : polyline) {
    enc.push(ll);
  }
  return std::move(enc.buf_);
}

template <int64_t Precision = 5>
geo::polyline decode_polyline(std::string_view const str) {
  constexpr auto const kPrecision = pow(10, Precision);
  int64_t lat{0};
  int64_t lng{0};

  auto const read = [](char const** first, char const* last) {
    int64_t buf{0};
    size_t shift = 0;
    while (*first != last) {
      int64_t curr = (**first) - 63;
      buf |= ((curr & kPolylineCurrMask) << shift);
      ++(*first);
      shift += kPolylineMaskBits;
      if ((curr & 0x20) == 0) {
        break;
      }
    }
    return (buf & 1) ? ~(buf >> 1) : (buf >> 1);
  };

  char const* first = str.data();
  char const* last = str.data() + str.size();

  geo::polyline polyline;
  while (first != last) {
    lat += read(&first, last);
    lng += read(&first, last);
    polyline.emplace_back(static_cast<double>(lat) / kPrecision,
                          static_cast<double>(lng) / kPrecision);
  }
  return polyline;
}

}  // namespace motis::path
