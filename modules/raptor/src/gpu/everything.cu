#include "motis/raptor/gpu/gpu_raptor.cuh"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/raptor_query.h"

#include <algorithm>

namespace motis::raptor {

__constant__ device_gpu_timetable GTT;

typedef time (*GetArrivalFun)(time const* const, stop_id const);
typedef bool (*UpdateArrivalFun)(time* const, stop_id const, time const);
typedef stop_id (*GetRouteStopFun)(route_stops_index const);
typedef time (*GetStopArrivalFun)(stop_times_index const);
typedef time (*GetStopDepartureFun)(stop_times_index const);

void cuda_sync_stream(cudaStream_t const& stream) {
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event,
                           cudaEventBlockingSync | cudaEventDisableTiming);
  cudaEventRecord(event, stream);
  cudaEventSynchronize(event);
  cudaEventDestroy(event);

  cc();
}

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args,
                          device_context const& device, cudaStream_t s) {
  cudaSetDevice(device.id_);

  cudaLaunchCooperativeKernel((void*)kernel, device.grid_,
                              device.threads_per_block_, args, 0, s);
  cc();
}

void fetch_result_from_device(d_query& dq) {
  //  auto& result = *dq.result_;

  // we do not need the last arrival array, it only exists because
  // how we calculate the footpaths
  cudaMemcpy(dq.host_.result_->data(), dq.device_.result_.front(),
             dq.host_.result_->byte_size(), cudaMemcpyDeviceToHost);
  cc();
  //  for (auto k = 0; k < max_raptor_round; ++k) {
  //    cudaMemcpy(result[k], dq.d_arrivals_[k], dq.stop_count_ * sizeof(time),
  //               cudaMemcpyDeviceToHost);
  //    cc();
  //  }
}

void fetch_arrivals_async(d_query const& dq, cudaStream_t s) {
  cudaMemcpyAsync(dq.host_.result_->data(), dq.device_.result_.front(),
                  dq.host_.result_->byte_size(), cudaMemcpyDeviceToHost, s);
  //  cudaMemcpyAsync((*dq.result_)[round_k], dq.d_arrivals_[round_k],
  //                  dq.stop_count_ * sizeof(time), cudaMemcpyDeviceToHost, s);
  cc();
}

void fetch_arrivals_async(d_query& dq, raptor_round const round_k,
                          cudaStream_t s) {
  //  cudaMemcpyAsync((*dq.result_)[round_k], dq.d_arrivals_[round_k],
  //                  dq.stop_count_ * sizeof(time), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(dq.host_.result_[round_k], dq.device_.result_[round_k],
                  dq.host_.result_->stop_count_ * sizeof(time),
                  cudaMemcpyDeviceToHost, s);
  //  cudaMemcpyAsync((*dq.result_)[round_k], dq.d_arrivals_[round_k],
  //                  dq.result_->stop_count_ * sizeof(time),
  //                  cudaMemcpyDeviceToHost, s);
  cc();
}

}  // namespace motis::raptor

#include "gpu_raptor.cu"
#include "gpu_timetable.cu"
#include "hybrid_raptor.cu"
