#pragma once

#include "motis/raptor/raptor_query.h"
#include "motis/raptor/gpu/memory_store.h"

namespace motis::raptor {

__device__ __forceinline__ unsigned int get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__ unsigned int get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__ unsigned int get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__ unsigned int get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args,
                          device_context const& device, cudaStream_t s,
                          raptor_criteria_config const criteria_config) {
  cudaSetDevice(device.id_);

  auto const& klc = device.launch_configs_.at(criteria_config);

  cudaLaunchCooperativeKernel((void*)kernel, klc.grid_,  //  NOLINT
                              klc.threads_per_block_, args, 0, s);
  cuda_check();
}

inline void fetch_statistics_async(d_query const& dq, cudaStream_t s) {
  cudaMemcpyAsync(dq.mem_->active_host_->stats_,
                  dq.mem_->active_device_->stats_,
                  sizeof(raptor_statistics),
                  cudaMemcpyDeviceToHost, s);
  cuda_check();
}

inline void fetch_arrivals_async(d_query const& dq, cudaStream_t s) {
  cudaMemcpyAsync(dq.mem_->active_host_->result_->data(),
                  dq.mem_->active_device_->result_.front(),
                  dq.mem_->active_host_->result_->byte_size(),
                  cudaMemcpyDeviceToHost, s);
  cuda_check();
}

inline void fetch_arrivals_async(d_query const& dq, raptor_round const round_k,
                                 cudaStream_t s) {
  cudaMemcpyAsync(
      (*dq.mem_->active_host_->result_)[round_k],
      dq.mem_->active_device_->result_[round_k],
      dq.mem_->active_host_->result_->arrival_times_count_ * sizeof(time),
      cudaMemcpyDeviceToHost, s);
  cuda_check();
}

}  // namespace motis::raptor