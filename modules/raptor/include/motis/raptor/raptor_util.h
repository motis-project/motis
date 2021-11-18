#pragma once

#include <algorithm>
#include <type_traits>

#if defined(MOTIS_CUDA)
#include "cuda_runtime.h"
#endif

namespace motis::raptor {

#if defined(MOTIS_CUDA)

#define __mark_cuda_rel__ \
  __host__ __device__

#endif

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
