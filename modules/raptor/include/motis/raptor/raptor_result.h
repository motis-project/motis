#pragma once

#include <cstddef>

#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include <array>
#include "cuda_runtime.h"
#endif

namespace motis::raptor {

struct raptor_result_base {
  raptor_result_base(stop_id const stop_count,
                     raptor_criteria_config const criteria_config)
      : arrival_times_count_{static_cast<arrival_id>(
            stop_count * get_trait_size_for_criteria_config(criteria_config))},
        criteria_config_{criteria_config} {}

  raptor_result_base() = delete;
  raptor_result_base(raptor_result_base const&) = delete;
  raptor_result_base(raptor_result_base const&&) = delete;
  raptor_result_base& operator=(raptor_result_base const&) = delete;
  raptor_result_base& operator=(raptor_result_base const&&) = delete;
  ~raptor_result_base() = default;

  time* operator[](raptor_round const index) {  // NOLINT
    return &result_[static_cast<ptrdiff_t>(index) * arrival_times_count_];
  };

  time const* operator[](raptor_round const index) const {
    return &result_[static_cast<ptrdiff_t>(index) * arrival_times_count_];
  };

  size_t byte_size() const {
    size_t const number_of_entries =
        static_cast<size_t>(max_raptor_round) * arrival_times_count_;
    size_t const size_in_bytes = sizeof(time) * number_of_entries;
    return size_in_bytes;
  }

  void reset() const {
    size_t const number_of_entries = byte_size() / sizeof(time);
    std::fill(result_, result_ + number_of_entries, invalid<time>);
  }

  time* data() const { return result_; }

  arrival_id arrival_times_count_{invalid<arrival_id>};
  raptor_criteria_config criteria_config_;
  time* result_{nullptr};
};

struct raptor_result : raptor_result_base {
  raptor_result(stop_id const stop_count,
                raptor_criteria_config const criteria_config)
      : raptor_result_base{stop_count, criteria_config} {
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
  raptor_result_pinned(stop_id stop_count,
                       raptor_criteria_config const criteria_config)
      : raptor_result_base{stop_count, criteria_config} {
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

}  // namespace motis::raptor