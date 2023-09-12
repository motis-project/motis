#pragma once

#include <cstddef>
#include <cstdint>

#include "ankerl/cista_adapter.h"

#include "cista/containers/array.h"
#include "cista/containers/mutable_fws_multimap.h"
#include "cista/containers/string.h"
#include "cista/containers/vector.h"
#include "cista/hash.h"

#include "geo/latlng.h"

namespace motis::footpaths {

template <typename K, typename V, typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = cista::raw::ankerl_map<K, V, Hash, Equality>;

template <typename K, typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using set = cista::raw::ankerl_set<K, Hash, Equality>;

template <typename K, typename V>
using mutable_fws_multimap = cista::raw::mutable_fws_multimap<K, V>;

template <typename V, std::size_t SIZE>
using array = cista::raw::array<V, SIZE>;

template <typename V>
using vector = cista::offset::vector<V>;

using nlocation_key_t = std::uint64_t;
using profile_key_t = std::uint8_t;

using string = cista::offset::string;
using strings = cista::offset::vector<string>;

// Returns a unique  nigiri location coordinate representation
// Interprets longitude and latitude as 32b float values. Appends them to each
// other and thus creates a 64b long coordinate representation.
// latitude (as 32b) || longitude (as 32b)
inline nlocation_key_t to_key(geo::latlng const& coord) {
  auto latf = static_cast<float>(coord.lat_);
  auto lngf = static_cast<float>(coord.lng_);

  auto latb = (*reinterpret_cast<std::uint32_t*>(&latf));
  auto lngb = (*reinterpret_cast<std::uint32_t*>(&lngf));

  return nlocation_key_t{static_cast<std::uint64_t>(latb) << 32 | lngb};
}

}  // namespace motis::footpaths
