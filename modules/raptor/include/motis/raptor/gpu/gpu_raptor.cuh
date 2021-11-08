#pragma once

#include "motis/raptor/raptor_query.h"

namespace motis::raptor {

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args,
                          device_context const& device, cudaStream_t s) {
  cudaSetDevice(device.id_);

  cudaLaunchCooperativeKernel((void*)kernel, device.grid_,
                              device.threads_per_block_, args, 0, s);
  cuda_check();
}

inline void fetch_arrivals_async(d_query const& dq, cudaStream_t s) {
  cudaMemcpyAsync(
      dq.mem_->host_.result_->data(), dq.mem_->device_.result_.front(),
      dq.mem_->host_.result_->byte_size(), cudaMemcpyDeviceToHost, s);
  cuda_check();
}

inline void fetch_arrivals_async(d_query const& dq, raptor_round const round_k,
                                 cudaStream_t s) {
  cudaMemcpyAsync((*dq.mem_->host_.result_)[round_k],
                  dq.mem_->device_.result_[round_k],
                  dq.mem_->host_.result_->stop_count_ * sizeof(time),
                  cudaMemcpyDeviceToHost, s);
  cuda_check();
}

__device__ void init_arrivals_dev(base_query const& query,
                                  device_memory const& device_mem,
                                  device_gpu_timetable const& tt);

__device__ void update_routes_dev(time const* prev_arrivals, time* arrivals,
                                  unsigned int* station_marks,
                                  unsigned int* route_marks,
                                  bool* any_station_marked,
                                  device_gpu_timetable const& tt);

__device__ void update_footpaths_dev(device_memory const& device_mem,
                                     raptor_round round_k,
                                     device_gpu_timetable const& tt);

void invoke_gpu_raptor(d_query const&);
void invoke_hybrid_raptor(d_query const&);

}  // namespace motis::raptor