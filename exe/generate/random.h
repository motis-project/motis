#pragma once

#include <atomic>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "nigiri/timetable.h"

#include "utl/verify.h"

namespace motis {

// shared by all threads and samplers of motis generate
inline std::atomic_uint32_t seed{0U};

inline std::uint32_t rand_in(std::uint32_t const from, std::uint32_t const to) {
  auto a = ++seed;
  a = (a ^ 61U) ^ (a >> 16U);
  a = a + (a << 3U);
  a = a ^ (a >> 4U);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15U);
  return from + (a % (to - from));
}

inline std::uint64_t rand_in(std::uint64_t const from, std::uint64_t const to) {
  auto const hi = rand_in(0U, std::numeric_limits<std::uint32_t>::max());
  auto const lo = rand_in(0U, std::numeric_limits<std::uint32_t>::max());
  auto const combined =
      (static_cast<std::uint64_t>(hi) << 32U) | static_cast<std::uint64_t>(lo);
  return from + (combined % (to - from));
}

// uniformly distributed in [0, 1): 53 random bits fill a double's mantissa
inline double rand_unit() {
  return static_cast<double>(
             rand_in(std::uint64_t{0U}, std::uint64_t{1U} << 53U)) *
         0x1.0p-53;
}

template <typename It>
It rand_in(It const begin, It const end) {
  return std::next(
      begin,
      rand_in(0U, static_cast<std::uint32_t>(std::distance(begin, end))));
}

template <typename Collection>
Collection::value_type rand_in(Collection const& c) {
  using std::begin;
  using std::end;
  utl::verify(!c.empty(), "empty collection");
  return *rand_in(begin(c), end(c));
}

inline nigiri::location_idx_t random_stop(
    nigiri::timetable const& tt,
    std::vector<nigiri::location_idx_t> const& stops) {
  auto s = nigiri::location_idx_t::invalid();
  do {
    s = rand_in(stops);
  } while (tt.location_routes_[s].empty());
  return s;
}

}  // namespace motis
