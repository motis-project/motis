#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include "cuda_runtime.h"

#include "utl/verify.h"
#include "utl/zip.h"

#include "motis/raptor/raptor_result.h"

namespace motis::raptor {

using device_id = int32_t;

inline auto get_launch_paramters(cudaDeviceProp const& prop,
                                 int32_t const concurrency_per_device) {
  int32_t block_dim_x = 32;  // must always be 32!
  int32_t block_dim_y = 32;  // range [1, ..., 32]
  int32_t block_size = block_dim_x * block_dim_y;
  int32_t max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;

  auto const mp_count = prop.multiProcessorCount / concurrency_per_device;

  int32_t num_blocks = mp_count * max_blocks_per_sm;

  dim3 threads_per_block(block_dim_x, block_dim_y, 1);
  dim3 grid(num_blocks, 1, 1);

  return std::make_tuple(threads_per_block, grid);
}

struct device_context {
  device_context() = delete;
  device_context(device_context const&) = delete;
  device_context(device_context const&&) = delete;
  device_context operator=(device_context const&) = delete;
  device_context operator=(device_context const&&) = delete;
  device_context(device_id const device_id,
                 int32_t const concurrency_per_device)
      : id_(device_id) {
    cudaSetDevice(id_);
    cuda_check();

    cudaGetDeviceProperties(&props_, device_id);
    cuda_check();

    std::tie(threads_per_block_, grid_) =
        get_launch_paramters(props_, concurrency_per_device);

    cudaStreamCreate(&proc_stream_);
    cuda_check();
    cudaStreamCreate(&transfer_stream_);
    cuda_check();
  }

  ~device_context() = default;

  void destroy() {
    cudaSetDevice(id_);
    cudaStreamDestroy(proc_stream_);
    cudaStreamDestroy(transfer_stream_);
    cuda_check();
  }

  device_id id_{};
  cudaDeviceProp props_{};

  dim3 threads_per_block_;
  dim3 grid_;

  cudaStream_t proc_stream_{};
  cudaStream_t transfer_stream_{};
};

struct host_memory {
  host_memory() = delete;
  host_memory(host_memory const&) = delete;
  host_memory(host_memory const&&) = delete;
  host_memory operator=(host_memory const&) = delete;
  host_memory operator=(host_memory const&&) = delete;
  explicit host_memory(stop_id const stop_count)
      : result_(new raptor_result_pinned(stop_count)) {
    cudaMallocHost(&any_station_marked_, sizeof(bool));

    *any_station_marked_ = false;
  }

  ~host_memory() = default;

  void destroy() {
    cudaFreeHost(any_station_marked_);
    delete result_;
  }

  void reset() const {
    *any_station_marked_ = false;
    result_->reset();
  }

  raptor_result_pinned* result_{nullptr};
  bool* any_station_marked_{nullptr};
};

struct device_memory {
  device_memory() = delete;
  device_memory(device_memory const&) = delete;
  device_memory(device_memory const&&) = delete;
  device_memory operator=(device_memory const&) = delete;
  device_memory operator=(device_memory const&&) = delete;
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
    cuda_check();

    this->reset_async(nullptr);
  }

  ~device_memory() = default;

  void destroy() {
    cudaFree(result_.front());
    cudaFree(footpaths_scratchpad_);
    cudaFree(station_marks_);
    cudaFree(route_marks_);
    cudaFree(any_station_marked_);
  }

  size_t get_result_bytes() const {
    return stop_count_ * sizeof(time) * max_raptor_round;
  }
  size_t get_station_mark_bytes() const { return ((stop_count_ / 32) + 1) * 4; }
  size_t get_route_mark_bytes() const { return ((route_count_ / 32) + 1) * 4; }
  size_t get_scratchpad_bytes() const { return stop_count_ * sizeof(time); }

  void reset_async(cudaStream_t s) const {
    cudaMemsetAsync(result_.front(), 0xFF, get_result_bytes(), s);
    cudaMemsetAsync(footpaths_scratchpad_, 0xFF, get_scratchpad_bytes(), s);
    cudaMemsetAsync(station_marks_, 0, get_station_mark_bytes(), s);
    cudaMemsetAsync(route_marks_, 0, get_route_mark_bytes(), s);
    cudaMemsetAsync(any_station_marked_, 0, sizeof(bool), s);
  }

  stop_id stop_count_{invalid<stop_id>};
  route_id route_count_{invalid<route_id>};

  device_result result_{};

  // TODO(julian) move from uint32_t to char or something
  uint32_t* route_marks_{};
  uint32_t* station_marks_{};
  bool* any_station_marked_{};
  time* footpaths_scratchpad_{};
};

struct mem {
  mem() = delete;
  mem(mem const&) = delete;
  mem(mem const&&) = delete;
  mem operator=(mem const&) = delete;
  mem operator=(mem const&&) = delete;

  mem(stop_id const stop_count, route_id const route_count,
      device_id const device_id, int32_t const concurrency_per_device)
      : host_(stop_count),
        device_(stop_count, route_count),
        context_(device_id, concurrency_per_device) {}

  ~mem() {
    host_.destroy();
    device_.destroy();
    context_.destroy();
  }

  host_memory host_;
  device_memory device_;
  device_context context_;
};

struct memory_store {
  using mem_idx = uint32_t;
  static_assert(std::is_unsigned_v<mem_idx>);

  void init(raptor_timetable const& tt, int32_t const concurrency_per_device) {
    int32_t device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (auto device_id = 0; device_id < device_count; ++device_id) {
      for (auto i = 0; i < concurrency_per_device; ++i) {
        memory_.emplace_back(
            std::make_unique<struct mem>(tt.stop_count(), tt.route_count(),
                                         device_id, concurrency_per_device));
      }
    }

    memory_mutexes_ = std::vector<std::mutex>(memory_.size());
  }

  mem_idx get_mem_idx() { return current_idx_.fetch_add(1) % memory_.size(); }

  std::atomic<mem_idx> current_idx_{0};
  static_assert(std::is_unsigned_v<decltype(current_idx_)::value_type>);

  std::vector<std::unique_ptr<mem>> memory_;
  std::vector<std::mutex> memory_mutexes_;
};

static_assert(
    std::is_unsigned_v<decltype(std::declval<memory_store>().get_mem_idx())>);

struct loaned_mem {
  loaned_mem() = delete;
  loaned_mem(loaned_mem const&) = delete;
  loaned_mem(loaned_mem const&&) = delete;
  loaned_mem operator=(loaned_mem const&) = delete;
  loaned_mem operator=(loaned_mem const&&) = delete;
  explicit loaned_mem(memory_store& store) {
    auto const idx = store.get_mem_idx();
    lock_ = std::unique_lock(store.memory_mutexes_[idx]);
    mem_ = store.memory_[idx].get();
  }

  ~loaned_mem() {
    mem_->device_.reset_async(mem_->context_.proc_stream_);
    mem_->host_.reset();
    cuda_sync_stream(mem_->context_.proc_stream_);
  }

  mem* mem_{nullptr};
  std::unique_lock<std::mutex> lock_{};
};

}  // namespace motis::raptor
