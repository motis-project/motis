#pragma once

#include "ankerl/cista_adapter.h"

#include "cista/containers/hash_map.h"
#include "cista/containers/string.h"

namespace motis::footpaths {

template <typename K, typename V, typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = cista::raw::ankerl_map<K, V, Hash, Equality>;

using string = cista::offset::string;
using strings = cista::offset::vector<string>;

}  // namespace motis::footpaths
