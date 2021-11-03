#include "motis/raptor/gpu/gpu_raptor.cuh"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/raptor_query.h"

#include <algorithm>

namespace motis::raptor {

__constant__ device_gpu_timetable GTT;

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args,
                          device_context const& device, cudaStream_t s) {
  cudaSetDevice(device.id_);

  cudaLaunchCooperativeKernel((void*)kernel, device.grid_,
                              device.threads_per_block_, args, 0, s);
  cc();
}

void fetch_arrivals_async(d_query const& dq, cudaStream_t s) {
  cudaMemcpyAsync(
      dq.mem_->host_.result_->data(), dq.mem_->device_.result_.front(),
      dq.mem_->host_.result_->byte_size(), cudaMemcpyDeviceToHost, s);
  cc();
}

void fetch_arrivals_async(d_query const& dq, raptor_round const round_k,
                          cudaStream_t s) {
  cudaMemcpyAsync((*dq.mem_->host_.result_)[round_k],
                  dq.mem_->device_.result_[round_k],
                  dq.mem_->host_.result_->stop_count_ * sizeof(time),
                  cudaMemcpyDeviceToHost, s);
  cc();
}

}  // namespace motis::raptor

#include "gpu_raptor.cu"
#include "gpu_timetable.cu"
#include "hybrid_raptor.cu"
