#pragma once

#include <atomic>
#include <thread>

#include "cuda_runtime.h"

#include "utl/verify.h"

#include "motis/raptor/raptor_result.h"

namespace motis::raptor {

struct device_context {
  int32_t id_;
  cudaDeviceProp props_;

  dim3 threads_per_block_;
  dim3 grid_;

  cudaStream_t proc_stream_;
  cudaStream_t transfer_stream_;
};

struct host_memory {
  host_memory() = delete;
  explicit host_memory(stop_id const stop_count) {
    cudaMallocHost(&any_station_marked_, sizeof(bool));
    *any_station_marked_ = false;

    result_ = new raptor_result_pinned(stop_count);
  }

  void free() {
    cudaFree(any_station_marked_);
    delete result_;
  }

  void reset() {
    *any_station_marked_ = false;
    result_->reset();
  }

  raptor_result_pinned* result_{nullptr};
  bool* any_station_marked_{nullptr};
};

struct device_memory {
  device_memory() = delete;

  device_memory(stop_id const stop_count, route_id const route_count)
      : stop_count_(stop_count), route_count_(route_count) {

    cudaMalloc(&(result_.front()), get_result_bytes());
    for (auto k = 1U; k < result_.size(); ++k) {
      result_[k] = result_[k - 1] + stop_count;
    }

    cudaMalloc(&footpaths_scratchpad_, get_scratchpad_bytes());
    cudaMalloc(&station_marks_, get_station_mark_bytes());
    cudaMalloc(&route_marks_, get_route_mark_bytes());
    cudaMalloc(&any_station_marked_, sizeof(bool));
  }

  size_t get_result_bytes() const {
    return stop_count_ * sizeof(time) * max_raptor_round;
  }
  size_t get_station_mark_bytes() const { return ((stop_count_ / 32) + 1) * 4; }
  size_t get_route_mark_bytes() const { return ((stop_count_ / 32) + 1) * 4; }
  size_t get_scratchpad_bytes() const { return stop_count_ * sizeof(time); }

  void free() {
    cudaFree(result_.front());
    cudaFree(footpaths_scratchpad_);
    cudaFree(station_marks_);
    cudaFree(route_marks_);
    cudaFree(any_station_marked_);
  }

  void reset_async(cudaStream_t s) {
    cudaMemsetAsync(result_.front(), 0xFF, get_result_bytes(), s);
    cudaMemsetAsync(footpaths_scratchpad_, 0xFF, get_scratchpad_bytes(), s);
    cudaMemsetAsync(station_marks_, 0, get_station_mark_bytes(), s);
    cudaMemsetAsync(route_marks_, 0, get_route_mark_bytes(), s);
    cudaMemsetAsync(any_station_marked_, 0, sizeof(bool), s);
  }

  //  device_gpu_timetable d_gtt_;
  stop_id stop_count_;
  route_id route_count_;

  device_result result_;

  // TODO(julian) move from uint32_t to char or something
  uint32_t* route_marks_;
  uint32_t* station_marks_;
  bool* any_station_marked_;
  time* footpaths_scratchpad_;
};

auto get_launch_paramters(cudaDeviceProp const& prop,
                          int32_t const concurrency_per_device) {
  int32_t block_dim_x = 32;  // must always be 32!
  int32_t block_dim_y = 32;  // range [1, ..., 32]
  int32_t block_size = block_dim_x * block_dim_y;
  int32_t max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;

  auto const mp_count =
      std::min(concurrency_per_device, prop.multiProcessorCount);

  int32_t num_blocks = mp_count * max_blocks_per_sm;

  dim3 threads_per_block(block_dim_x, block_dim_y, 1);
  dim3 grid(num_blocks, 1, 1);

  return std::make_tuple(threads_per_block, grid);
}

std::vector<device_context> get_device_contexts(
    int32_t concurrency_per_device) {
  std::vector<device_context> contexts;

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  cc();

  for (int device_id = 0; device_id < device_count; ++device_id) {
    cudaSetDevice(device_id);
    cc();

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);
    cc();

    auto const& [threads_per_block, grid] = get_launch_paramters(
        prop, static_cast<int32_t>(concurrency_per_device));

    for (auto i = 0; i < concurrency_per_device; ++i) {
      device_context dc;

      dc.id_ = device_id;
      dc.props_ = prop;

      dc.threads_per_block_ = threads_per_block;
      dc.grid_ = grid;

      cudaStreamCreate(&dc.proc_stream_);
      cc();
      cudaStreamCreate(&dc.transfer_stream_);
      cc();
    }
  }

  return contexts;
}

struct memory_store {
  using mem_idx = uint32_t;
  static_assert(std::is_unsigned_v<mem_idx>);

  void init(raptor_timetable const& tt, uint32_t const concurrency_per_device) {
    int32_t device_count = 0;
    cudaGetDeviceCount(&device_count);

    concurrency_per_device_ = concurrency_per_device;
    max_concurrency_ = concurrency_per_device_ * device_count;

    device_ = std::vector<device_memory>(
        max_concurrency_, device_memory(tt.stop_count(), tt.route_count()));
    host_ = std::vector<host_memory>(max_concurrency_,
                                     host_memory(tt.stop_count()));

    context_ =
        get_device_contexts(static_cast<int32_t>(concurrency_per_device_));

    utl::verify(device_.size() == host_.size(),
                "Memory stores do not have the same size!");
    utl::verify(host_.size() == context_.size(),
                "Memory stores do not have the same size!");
  }

  mem_idx get_mem_idx() { return current_idx_.fetch_add(1) % max_concurrency_; }

  mem_idx max_concurrency_{0};
  mem_idx concurrency_per_device_{0};

  std::atomic<mem_idx> current_idx_{0};
  static_assert(std::is_unsigned_v<decltype(current_idx_)::value_type>);

  std::vector<device_memory> device_;
  std::vector<host_memory> host_;
  std::vector<device_context> context_;
};

static_assert(
    std::is_unsigned_v<decltype(std::declval<memory_store>().get_mem_idx())>);

}  // namespace motis::raptor
