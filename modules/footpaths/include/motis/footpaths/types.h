#pragma once

#include "cista/containers/hash_map.h"
#include "cista/containers/string.h"

#include "nigiri/types.h"

namespace motis::footpaths {

template <typename K, typename V, typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = cista::raw::ankerl_map<K, V, Hash, Equality>;

using string = cista::raw::string;
}  // namespace motis::footpaths