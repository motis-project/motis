#pragma once

#include <algorithm>
#include <type_traits>

namespace motis::raptor {

template <typename T>
inline auto vec_size_bytes(std::vector<T> const& vec) {
  static_assert(std::is_trivially_copyable_v<T>);
  return sizeof(T) * vec.size();
}

template <typename Container, typename T>
inline bool contains(Container const& c, T const& ele) {
  return std::find(std::begin(c), std::end(c), ele) != std::end(c);
}

}  // namespace motis::raptor
