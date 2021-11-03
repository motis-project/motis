#pragma once

#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "cuda_runtime.h"
#endif

namespace motis::raptor {

struct raptor_result_base {
  explicit raptor_result_base(stop_id const stop_count)
      : stop_count_(stop_count) {}

  raptor_result_base() = delete;
  raptor_result_base(raptor_result_base const&) = delete;
  raptor_result_base(raptor_result_base const&&) = delete;
  raptor_result_base& operator=(raptor_result_base const&) = delete;
  raptor_result_base& operator=(raptor_result_base const&&) = delete;

  const time* operator[](raptor_round const index) const {
    return &result_[index * stop_count_];
  };

  time* operator[](raptor_round const index) {
    return &result_[index * stop_count_];
  };

  size_t byte_size() const {
    size_t const number_of_entries = max_raptor_round * stop_count_;
    size_t const size_in_bytes = sizeof(time) * number_of_entries;
    return size_in_bytes;
  }

  void reset() {
    size_t const number_of_entries = byte_size() / sizeof(time);
    std::fill(result_, result_ + number_of_entries, invalid<time>);
  }

  time* data() { return result_; }

  stop_id stop_count_;
  time* result_;
};

struct raptor_result : raptor_result_base {
  raptor_result(stop_id const stop_count) : raptor_result_base(stop_count) {
    result_ = new time[this->byte_size()];
    this->reset();
  }

  ~raptor_result() { delete[] result_; };
};

#if defined(MOTIS_CUDA)
using device_result = std::array<time*, max_raptor_round>;

struct raptor_result_pinned : raptor_result_base {
  raptor_result_pinned(stop_id stop_count) : raptor_result_base(stop_count) {
    cudaMallocHost(&result_, this->byte_size());
    this->reset();
  }

  ~raptor_result_pinned() { cudaFreeHost(result_); };
};
#endif

}  // namespace motis::raptor