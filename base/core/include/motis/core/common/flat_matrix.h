#pragma once

#include <cinttypes>

#include "motis/vector.h"

namespace motis {

template <typename T>
struct flat_matrix {
  struct row {
    row(flat_matrix& matrix, int row_index)
        : matrix_(matrix), row_index_(row_index) {}

    T& operator[](int column_index) {
      auto pos = matrix_.column_count_ * row_index_ + column_index;
      return matrix_.entries_[pos];
    }

    flat_matrix& matrix_;
    int row_index_;
  };

  struct const_row {
    const_row(flat_matrix const& matrix, int row_index)
        : matrix_(matrix), row_index_(row_index) {}

    T const& operator[](int column_index) const {
      auto pos = matrix_.column_count_ * row_index_ + column_index;
      return matrix_.entries_[pos];
    }

    flat_matrix const& matrix_;
    int row_index_;
  };

  row operator[](int row_index) { return {*this, row_index}; }
  const_row operator[](int row_index) const { return {*this, row_index}; }

  T& operator()(int const row_index, int const column_index) {
    return entries_[column_count_ * row_index + column_index];
  }

  uint32_t column_count_{0U};
  mcd::vector<T> entries_;
};

template <typename T>
inline flat_matrix<T> make_flat_matrix(uint32_t const column_count,
                                       T const& init = T{}) {
  auto v = mcd::vector<T>{};
  v.resize(column_count * column_count, init);
  return flat_matrix<T>{column_count, std::move(v)};
}

}  // namespace motis
