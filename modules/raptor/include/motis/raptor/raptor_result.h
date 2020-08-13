#pragma once

#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

inline time* alloc_result_memory(size_t const size_in_bytes) {
  return new time[size_in_bytes];
}

inline void delete_result_memory(const time* result) { delete[] result; }

// time* alloc_pinned_memory(size_t const) {
//  time* result = nullptr;
//  cudaMallocHost(&result, size_in_bytes);
//  return result;
//}

// void free_pinned_memory(time*) { cudaFreeHost(result); }

template <decltype(alloc_result_memory) Alloc,
          decltype(delete_result_memory) Free>
struct raptor_result_gen {
  raptor_result_gen() = delete;
  raptor_result_gen(raptor_result_gen const&) = delete;
  raptor_result_gen(raptor_result_gen const&&) = delete;
  raptor_result_gen& operator=(raptor_result_gen const&) = delete;
  raptor_result_gen& operator=(raptor_result_gen const&&) = delete;

  explicit raptor_result_gen(station_id const stop_count)
      : stop_count_(stop_count) {
    size_t const number_of_entries = max_raptor_round * stop_count_;
    size_t const size_in_bytes = sizeof(time) * number_of_entries;

    result_ = Alloc(size_in_bytes);
    std::fill(result_, result_ + number_of_entries, invalid<time>);
  }

  ~raptor_result_gen() { Free(result_); };

  const time* operator[](raptor_round const index) const {
    return &result_[index * stop_count_];
  };

  time* operator[](raptor_round const index) {
    return &result_[index * stop_count_];
  };

private:
  station_id stop_count_;
  time* result_;
};

using raptor_result =
    raptor_result_gen<alloc_result_memory, delete_result_memory>;

// using raptor_result_pinned =
//    raptor_result_gen<alloc_pinned_memory, free_pinned_memory>;

}  // namespace motis::raptor