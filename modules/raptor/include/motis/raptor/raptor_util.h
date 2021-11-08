#pragma once

#include <algorithm>
#include <functional>

namespace motis::raptor {

template <typename T>
inline auto vec_size_bytes(std::vector<T> const& vec) {
  static_assert(std::is_trivially_copyable_v<T>);
  return sizeof(T) * vec.size();
}

// append contents of a vector to another one
template <typename T>
inline void append_vector(std::vector<T>& dst, std::vector<T> const& elems) {
  dst.insert(std::end(dst), std::begin(elems), std::end(elems));
}

template <typename Container, typename T>
inline bool contains(Container const& c, T const& ele) {
  return std::find(std::begin(c), std::end(c), ele) != std::end(c);
}

}  // namespace motis::raptor
