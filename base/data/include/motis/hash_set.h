#pragma once

#if defined(MOTIS_USE_STD)

#include <unordered_set>

namespace mcd {

template <typename T, typename Hash, typename Eq>
using hash_set = std::unordered_set<T, Hash, Eq>;

}  // namespace mcd

#else

#include "cista/containers/hash_set.h"

#include "motis/data.h"

namespace mcd {

using motis::data::hash_set;

}  // namespace mcd

#endif

namespace mcd {

template <typename Entry, typename Hash, typename Eq, typename CreateFun>
auto set_get_or_create(hash_set<Entry, Hash, Eq>& s, Entry const& key,
                       CreateFun f) -> decltype(*s.find(key))& {
  auto it = s.find(key);
  if (it != s.end()) {
    return *it;
  } else {
    return *s.insert(f()).first;
  }
}

}  // namespace mcd
