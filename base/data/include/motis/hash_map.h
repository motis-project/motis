#pragma once

#if defined(MOTIS_USE_STD)

#include <unordered_map>

namespace mcd {

template <typename T, typename Hash, typename Eq>
using hash_map = std::unordered_map<T, Hash, Eq>;

}  // namespace mcd

#else

#include "cista/containers/hash_map.h"

#include "motis/data.h"

namespace mcd {

using motis::data::hash_map;

}  // namespace mcd

#endif