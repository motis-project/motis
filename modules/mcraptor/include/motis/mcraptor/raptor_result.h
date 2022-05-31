#pragma once

#include <cstddef>

#include "motis/mcraptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "cuda_runtime.h"
#endif

namespace motis::mcraptor {

struct raptor_result_base {
  explicit raptor_result_base(stop_id const stop_count)
      : stop_count_{stop_count} {}

  raptor_result_base() = delete;
  raptor_result_base(raptor_result_base const&) = delete;
  raptor_result_base(raptor_result_base const&&) = delete;
  raptor_result_base& operator=(raptor_result_base const&) = delete;
  raptor_result_base& operator=(raptor_result_base const&&) = delete;
  ~raptor_result_base() = default;

  time* operator[](raptor_round const index) {  // NOLINT
    return &result_[static_cast<ptrdiff_t>(index) * stop_count_];
  };

  time const* operator[](raptor_round const index) const {
    return &result_[static_cast<ptrdiff_t>(index) * stop_count_];
  };

  size_t byte_size() const {
    size_t const number_of_entries =
        static_cast<size_t>(max_raptor_round) * stop_count_;
    size_t const size_in_bytes = sizeof(time) * number_of_entries;
    return size_in_bytes;
  }

  void reset() const {
    size_t const number_of_entries = byte_size() / sizeof(time);
    std::fill(result_, result_ + number_of_entries, invalid<time>);
  }

  time* data() const { return result_; }

  stop_id stop_count_{invalid<stop_id>};
  time* result_{nullptr};
};

struct raptor_result : raptor_result_base {
  explicit raptor_result(stop_id const stop_count)
      : raptor_result_base{stop_count} {
    result_ = new time[this->byte_size()];
    this->reset();
  }
  raptor_result() = delete;
  raptor_result(raptor_result const&) = delete;
  raptor_result(raptor_result const&&) = delete;
  raptor_result& operator=(raptor_result const&) = delete;
  raptor_result& operator=(raptor_result const&&) = delete;

  ~raptor_result() { delete[] result_; };
};

#if defined(MOTIS_CUDA)
using device_result = std::array<time*, max_raptor_round>;

struct raptor_result_pinned : public raptor_result_base {
  explicit raptor_result_pinned(stop_id stop_count)
      : raptor_result_base{stop_count} {
    cudaMallocHost(&result_, this->byte_size());
    this->reset();
  }

  raptor_result_pinned() = delete;
  raptor_result_pinned(raptor_result_pinned const&) = delete;
  raptor_result_pinned(raptor_result_pinned const&&) = delete;
  raptor_result_pinned& operator=(raptor_result_pinned const&) = delete;
  raptor_result_pinned& operator=(raptor_result_pinned const&&) = delete;

  ~raptor_result_pinned() { cudaFreeHost(result_); };
};
#endif

}  // namespace motis::mcraptor