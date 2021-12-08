#include "motis/raptor/gpu/gpu_raptor.cuh"

#include "motis/raptor/gpu/raptor_utils.cuh"

namespace motis::raptor {

__global__ void init_arrivals_kernel(base_query const query,
                                     device_memory const device_mem,
                                     device_gpu_timetable const tt) {
  init_arrivals_dev(query, device_mem, tt);
}

__global__ void update_footpaths_kernel(device_memory const device_mem,
                                        raptor_round round_k,
                                        device_gpu_timetable const tt) {
  update_footpaths_dev(device_mem, round_k, tt);
}

__global__ void update_routes_kernel(device_memory const device_mem,
                                     raptor_round round_k,
                                     device_gpu_timetable const tt) {
  time const* const prev_arrivals = device_mem.result_[round_k - 1];
  time* const arrivals = device_mem.result_[round_k];

  update_routes_dev(prev_arrivals, arrivals, device_mem.station_marks_,
                    device_mem.route_marks_, tt);
}

void invoke_hybrid_raptor(d_query const& dq) {
  auto const& proc_stream = dq.mem_->context_.proc_stream_;
  auto const& transfer_stream = dq.mem_->context_.transfer_stream_;

  void* init_args[] = {(void*)&dq, (void*)dq.mem_->active_device_,  // NOLINT
                       (void*)&dq.tt_};  // NOLINT

  launch_kernel(init_arrivals_kernel, init_args, dq.mem_->context_, proc_stream,
                raptor_criteria_config::Default);
  cuda_sync_stream(proc_stream);

  fetch_arrivals_async(dq, 0, transfer_stream);

  for (int k = 1; k < max_raptor_round; ++k) {
    void* kernel_args[] = {(void*)dq.mem_->active_device_, (void*)&k,  // NOLINT
                           (void*)&dq.tt_};  // NOLINT

    launch_kernel(update_routes_kernel, kernel_args, dq.mem_->context_,
                  proc_stream, raptor_criteria_config::Default);
    cuda_sync_stream(proc_stream);

    launch_kernel(update_footpaths_kernel, kernel_args, dq.mem_->context_,
                  proc_stream, raptor_criteria_config::Default);
    cuda_sync_stream(proc_stream);

    cudaMemcpyAsync(dq.mem_->active_host_->any_station_marked_,
                    dq.mem_->active_device_->overall_station_marked_,
                    sizeof(bool), cudaMemcpyDeviceToHost, transfer_stream);
    cuda_sync_stream(transfer_stream);

    if (!*dq.mem_->active_host_->any_station_marked_) {
      break;
    }

    fetch_arrivals_async(dq, k, transfer_stream);
  }

  cuda_sync_stream(transfer_stream);
}

}  // namespace motis::raptor