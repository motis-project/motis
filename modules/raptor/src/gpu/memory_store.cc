#ifdef MOTIS_CUDA
#include "motis/raptor/gpu/memory_store.h"

#include "cuda_runtime.h"

#include "motis/raptor/gpu/cuda_util.h"

namespace motis::raptor {

std::pair<dim3, dim3> get_launch_paramters(
    cudaDeviceProp const& prop, int32_t const concurrency_per_device) {
  int32_t block_dim_x = 32;  // must always be 32!
  int32_t block_dim_y = 32;  // range [1, ..., 32]
  int32_t block_size = block_dim_x * block_dim_y;
  int32_t max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;

  auto const mp_count = prop.multiProcessorCount / concurrency_per_device;

  int32_t num_blocks = mp_count * max_blocks_per_sm;

  dim3 threads_per_block(block_dim_x, block_dim_y, 1);
  dim3 grid(num_blocks, 1, 1);

  return {threads_per_block, grid};
}

device_context::device_context(device_id const device_id,
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

void device_context::destroy() {
  cudaSetDevice(id_);
  cudaStreamDestroy(proc_stream_);
  proc_stream_ = cudaStream_t{};
  cudaStreamDestroy(transfer_stream_);
  transfer_stream_ = cudaStream_t{};
  cuda_check();
}

host_memory::host_memory(stop_id const stop_count)
    : result_{std::make_unique<raptor_result_pinned>(stop_count)} {
  cudaMallocHost(&any_station_marked_, sizeof(bool));
  *any_station_marked_ = false;
}

void host_memory::destroy() {
  cudaFreeHost(any_station_marked_);
  any_station_marked_ = nullptr;
  result_ = nullptr;
}

void host_memory::reset() const {
  *any_station_marked_ = false;
  result_->reset();
}

device_memory::device_memory(stop_id const stop_count,
                             route_id const route_count,
                             size_t const max_add_starts)
    : stop_count_{stop_count},
      route_count_{route_count},
      max_add_starts_{max_add_starts} {
  cudaMalloc(&(result_.front()), get_result_bytes());
  for (auto k = 1U; k < result_.size(); ++k) {
    result_[k] = result_[k - 1] + stop_count;
  }

  cudaMalloc(&footpaths_scratchpad_, get_scratchpad_bytes());
  cudaMalloc(&station_marks_, get_station_mark_bytes());
  cudaMalloc(&route_marks_, get_route_mark_bytes());
  cudaMalloc(&any_station_marked_, sizeof(bool));
  cudaMalloc(&additional_starts_, get_additional_starts_bytes());
  cuda_check();

  this->reset_async(nullptr);
}

void device_memory::destroy() {
  cudaFree(result_.front());
  cudaFree(footpaths_scratchpad_);
  cudaFree(station_marks_);
  cudaFree(route_marks_);
  cudaFree(any_station_marked_);
  cudaFree(additional_starts_);
}

size_t device_memory::get_result_bytes() const {
  return stop_count_ * sizeof(time) * max_raptor_round;
}

size_t device_memory::get_station_mark_bytes() const {
  return ((stop_count_ / 32) + 1) * 4;
}

size_t device_memory::get_route_mark_bytes() const {
  return ((route_count_ / 32) + 1) * 4;
}

size_t device_memory::get_scratchpad_bytes() const {
  return stop_count_ * sizeof(time);
}

size_t device_memory::get_additional_starts_bytes() const {
  return max_add_starts_ * sizeof(additional_start);
}

void device_memory::reset_async(cudaStream_t s) {
  cudaMemsetAsync(result_.front(), 0xFF, get_result_bytes(), s);
  cudaMemsetAsync(footpaths_scratchpad_, 0xFF, get_scratchpad_bytes(), s);
  cudaMemsetAsync(station_marks_, 0, get_station_mark_bytes(), s);
  cudaMemsetAsync(route_marks_, 0, get_route_mark_bytes(), s);
  cudaMemsetAsync(any_station_marked_, 0, sizeof(bool), s);
  cudaMemsetAsync(additional_starts_, 0xFF, get_additional_starts_bytes(), s);
  additional_start_count_ = invalid<decltype(additional_start_count_)>;
}

mem::mem(stop_id const stop_count, route_id const route_count,
         size_t const max_add_starts, device_id const device_id,
         int32_t const concurrency_per_device)
    : host_{stop_count},
      device_{stop_count, route_count, max_add_starts},
      context_{device_id, concurrency_per_device} {}

mem::~mem() {
  host_.destroy();
  device_.destroy();
  context_.destroy();
}

void memory_store::init(raptor_meta_info const& meta_info,
                        raptor_timetable const& tt,
                        int32_t const concurrency_per_device) {
  int32_t device_count = 0;
  cudaGetDeviceCount(&device_count);

  auto const max_add_starts = get_max_add_starts(meta_info);

  for (auto device_id = 0; device_id < device_count; ++device_id) {
    for (auto i = 0; i < concurrency_per_device; ++i) {
      memory_.emplace_back(std::make_unique<struct mem>(
          tt.stop_count(), tt.route_count(), max_add_starts, device_id,
          concurrency_per_device));
    }
  }

  memory_mutexes_ = std::vector<std::mutex>(memory_.size());
}

memory_store::mem_idx memory_store::get_mem_idx() {
  return current_idx_.fetch_add(1) % memory_.size();
}

loaned_mem::loaned_mem(memory_store& store) {
  auto const idx = store.get_mem_idx();
  lock_ = std::unique_lock(store.memory_mutexes_[idx]);
  mem_ = store.memory_[idx].get();
}

loaned_mem::~loaned_mem() {
  mem_->device_.reset_async(mem_->context_.proc_stream_);
  mem_->host_.reset();
  cuda_sync_stream(mem_->context_.proc_stream_);
}

}  // namespace motis::raptor
#endif