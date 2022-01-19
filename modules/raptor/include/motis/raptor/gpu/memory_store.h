#pragma once

#include <atomic>
#include <mutex>

#include "motis/raptor/additional_start.h"
#include "motis/raptor/raptor_result.h"

namespace motis::raptor {

using device_id = int32_t;

std::pair<dim3, dim3> get_launch_paramters(cudaDeviceProp const& prop,
                                           int32_t concurrency_per_device);

struct device_context {
  device_context() = delete;
  device_context(device_context const&) = delete;
  device_context(device_context const&&) = delete;
  device_context operator=(device_context const&) = delete;
  device_context operator=(device_context const&&) = delete;
  device_context(device_id device_id, int32_t concurrency_per_device);

  ~device_context() = default;

  void destroy();

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
  explicit host_memory(stop_id stop_count);

  ~host_memory() = default;

  void destroy();
  void reset() const;

  std::unique_ptr<raptor_result_pinned> result_{nullptr};
  bool* any_station_marked_{nullptr};
};

struct device_memory {
  device_memory() = delete;
  device_memory(device_memory const&) = delete;
  device_memory(device_memory const&&) = delete;
  device_memory operator=(device_memory const&) = delete;
  device_memory operator=(device_memory const&&) = delete;
  device_memory(stop_id stop_count, route_id route_count,
                size_t max_add_starts);

  ~device_memory() = default;

  void destroy();

  size_t get_result_bytes() const;
  size_t get_station_mark_bytes() const;
  size_t get_route_mark_bytes() const;
  size_t get_scratchpad_bytes() const;
  size_t get_additional_starts_bytes() const;

  void reset_async(cudaStream_t s);

  stop_id stop_count_{invalid<stop_id>};
  route_id route_count_{invalid<route_id>};
  size_t max_add_starts_{invalid<size_t>};

  device_result result_{};

  // TODO(julian) move from uint32_t to char or something
  uint32_t* route_marks_{};
  uint32_t* station_marks_{};
  bool* any_station_marked_{};
  time* footpaths_scratchpad_{};
  additional_start* additional_starts_{};
  size_t additional_start_count_{};
};

struct mem {
  mem() = delete;
  mem(mem const&) = delete;
  mem(mem const&&) = delete;
  mem operator=(mem const&) = delete;
  mem operator=(mem const&&) = delete;

  mem(stop_id stop_count, route_id route_count, size_t max_add_starts,
      device_id device_id, int32_t concurrency_per_device);

  ~mem();

  host_memory host_;
  device_memory device_;
  device_context context_;
};

struct memory_store {
  using mem_idx = uint32_t;
  static_assert(std::is_unsigned_v<mem_idx>);

  void init(raptor_meta_info const& meta_info, raptor_timetable const& tt,
            int32_t concurrency_per_device);

  mem_idx get_mem_idx();

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
  explicit loaned_mem(memory_store& store);

  ~loaned_mem();

  mem* mem_{nullptr};
  std::unique_lock<std::mutex> lock_{};
};

}  // namespace motis::raptor
