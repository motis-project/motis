#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>

#include "motis/raptor/additional_start.h"
#include "motis/raptor/criteria/configs.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_statistics.h"

namespace motis::raptor {

using device_id = int32_t;

struct kernel_launch_config {
  dim3 threads_per_block_;
  dim3 grid_;
};

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

  std::unordered_map<raptor_criteria_config, kernel_launch_config>
      launch_configs_;

  cudaStream_t proc_stream_{};
  cudaStream_t transfer_stream_{};
};

struct host_memory {
  host_memory() = delete;
  host_memory(host_memory const&) = delete;
  host_memory(host_memory const&&) = delete;
  host_memory operator=(host_memory const&) = delete;
  host_memory operator=(host_memory const&&) = delete;
  host_memory(stop_id stop_count, raptor_criteria_config criteria_config);

  ~host_memory() = default;

  void destroy();
  void reset() const;

  std::unique_ptr<raptor_result_pinned> result_{};
  raptor_statistics* stats_{nullptr};
  bool* any_station_marked_{nullptr};
};

struct device_memory {
  device_memory() = delete;
  device_memory(device_memory const&) = delete;
  device_memory(device_memory const&&) = delete;
  device_memory operator=(device_memory const&) = delete;
  device_memory operator=(device_memory const&&) = delete;
  device_memory(stop_id stop_count, raptor_criteria_config criteria_config,
                route_id route_count, size_t max_add_starts);

  ~device_memory() = default;

  void destroy();

  size_t get_result_bytes() const;
  size_t get_station_mark_bytes() const;
  size_t get_route_mark_bytes() const;
  size_t get_scratchpad_bytes() const;
  size_t get_additional_starts_bytes() const;
  size_t get_fp_mark_bytes() const;

  void reset_async(cudaStream_t s);

  stop_id stop_count_{invalid<stop_id>};
  route_id route_count_{invalid<route_id>};
  size_t max_add_starts_{invalid<size_t>};
  arrival_id arrival_times_count_{invalid<arrival_id>};
  trait_id trait_size_{invalid<trait_id>};
  size_t additional_start_count_{0};

  device_result result_{};
  uint32_t* route_marks_{nullptr};
  uint32_t* station_marks_{nullptr};
  bool* any_station_marked_{nullptr};
  bool* overall_station_marked_{nullptr};
  time* footpaths_scratchpad_{nullptr};
  time* earliest_arrivals_{nullptr};
  additional_start* additional_starts_{nullptr};
  raptor_statistics* stats_{nullptr};
  uint32_t* fp_marks_{nullptr};
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

  void reset_active();
  void require_active(raptor_criteria_config criteria_config);

  device_context context_;
  // host_memory host_;
  // device_memory device_;

  host_memory* active_host_;
  device_memory* active_device_;

private:
  raptor_criteria_config active_config_;
  bool is_reset_;
  std::unordered_map<raptor_criteria_config, std::unique_ptr<host_memory>>
      host_memories_;
  std::unordered_map<raptor_criteria_config, std::unique_ptr<device_memory>>
      device_memories_;
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
