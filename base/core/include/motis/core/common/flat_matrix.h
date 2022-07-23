#pragma once

#include <cinttypes>

#include "motis/vector.h"

namespace motis {

template <typename Vector>
struct flat_matrix {
  using value_type = typename Vector::value_type;

  struct row {
    row(flat_matrix& matrix, int row_index)
        : matrix_(matrix), row_index_(row_index) {}

    value_type& operator[](int column_index) {
      auto pos = matrix_.column_count_ * row_index_ + column_index;
      return matrix_.entries_[pos];
    }

    flat_matrix& matrix_;
    int row_index_;
  };

  struct const_row {
    const_row(flat_matrix const& matrix, int row_index)
        : matrix_(matrix), row_index_(row_index) {}

    value_type const& operator[](int column_index) const {
      auto pos = matrix_.column_count_ * row_index_ + column_index;
      return matrix_.entries_[pos];
    }

    flat_matrix const& matrix_;
    int row_index_;
  };

  row operator[](int row_index) { return {*this, row_index}; }
  const_row operator[](int row_index) const { return {*this, row_index}; }

  value_type& operator()(int const row_index, int const column_index) {
    return entries_[column_count_ * row_index + column_index];
  }

  uint32_t column_count_{0U};
  Vector entries_;
};

template <typename T>
inline flat_matrix<mcd::vector<T>> make_flat_matrix(uint32_t const column_count,
                                                    T const& init = T{}) {
  auto v = mcd::vector<T>{};
  v.resize(column_count * column_count, init);
  return flat_matrix<mcd::vector<T>>{column_count, std::move(v)};
}

template <typename T>
inline flat_matrix<std::vector<T>> make_std_flat_matrix(
    uint32_t const column_count, T const& init = T{}) {
  auto v = std::vector<T>{};
  v.resize(column_count * column_count, init);
  return flat_matrix<std::vector<T>>{column_count, std::move(v)};
}

}  // namespace motis
