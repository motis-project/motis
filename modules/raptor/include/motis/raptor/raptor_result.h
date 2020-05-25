#pragma once

#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

typedef time* (*AllocResultMemoryFun)(size_t const);
typedef void (*FreeResultMemoryFun)(time*);

template <AllocResultMemoryFun Alloc, FreeResultMemoryFun Free>
struct raptor_result_gen {
  raptor_result_gen() = delete;
  raptor_result_gen(raptor_result_gen const&) = delete;
  raptor_result_gen& operator=(raptor_result_gen const&) = delete;
  raptor_result_gen(raptor_result_gen const&&) = delete;
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

time* alloc_result_memory(size_t const size_in_bytes) {
  return (time*)(malloc(size_in_bytes));
}

void free_result_memory(time* result) { free(result); }

using raptor_result =
    raptor_result_gen<alloc_result_memory, free_result_memory>;

// time* alloc_pinned_memory(size_t const) {
//  time* result = nullptr;
//  cudaMallocHost(&result, size_in_bytes);
//  return result;
//}

// void free_pinned_memory(time*) { cudaFreeHost(result); }

// using raptor_result_pinned =
//    raptor_result_gen<alloc_pinned_memory, free_pinned_memory>;

}  // namespace motis::raptor