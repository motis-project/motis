#pragma once

#include "utl/verify.h"

#include "motis/core/common/flat_matrix.h"

namespace motis {

template <typename T>
void floyd_warshall(T&& mat) {
  utl::verify(mat.entries_.size() == mat.column_count_ * mat.column_count_,
              "floyd_warshall: input is not a square matrix.");
  using value_type = typename std::decay_t<T>::value_type;
  constexpr uint64_t const kMaxDistance =
      std::numeric_limits<value_type>::max();

  for (auto k = 0UL; k < mat.column_count_; ++k) {
    for (auto i = 0UL; i < mat.column_count_; ++i) {
      for (auto j = 0UL; j < mat.column_count_; ++j) {
        auto const distance = static_cast<value_type>(
            std::min(kMaxDistance, static_cast<uint64_t>(mat(i, k)) +
                                       static_cast<uint64_t>(mat(k, j))));
        if (mat(i, j) > distance) {
          mat(i, j) = distance;
        }
      }
    }
  }
}

}  // namespace motis
