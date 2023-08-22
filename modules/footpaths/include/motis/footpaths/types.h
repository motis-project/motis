#pragma once

#include <cstdint>

#include "ankerl/cista_adapter.h"

#include "cista/containers/string.h"
#include "cista/containers/vector.h"

namespace motis::footpaths {

template <typename K, typename V, typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = cista::raw::ankerl_map<K, V, Hash, Equality>;

template <typename K, typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using set = cista::raw::ankerl_set<K, Hash, Equality>;

template <typename V>
using vector = cista::offset::vector<V>;

using key64_t = std::uint64_t;

using string = cista::offset::string;
using strings = cista::offset::vector<string>;

}  // namespace motis::footpaths
