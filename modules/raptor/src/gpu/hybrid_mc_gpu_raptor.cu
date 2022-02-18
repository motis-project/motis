#include "motis/raptor/gpu/mc_gpu_raptor.cuh"

#include "motis/raptor/gpu/raptor_utils.cuh"

namespace motis::raptor {

template <typename CriteriaConfig>
__global__ void mc_init_arrivals_kernel(base_query const query,
                                        device_memory const device_mem,
                                        device_gpu_timetable const tt) {
  mc_init_arrivals_dev<CriteriaConfig>(query, device_mem, tt);
}

template <typename CriteriaConfig>
__global__ void mc_update_footpaths_kernel(device_memory const device_mem,
                                           raptor_round const round_k,
                                           stop_id const target_stop_id,
                                           device_gpu_timetable const tt) {
  mc_update_footpaths_dev<CriteriaConfig>(device_mem, round_k, tt);
}

template <typename CriteriaConfig>
__global__ void mc_update_routes_kernel(device_memory const device_mem,
                                        raptor_round round_k,
                                        stop_id const target_stop_id,
                                        device_gpu_timetable const tt) {
  mc_update_routes_dev<CriteriaConfig>(device_mem, round_k, target_stop_id, tt);
}

template <typename CriteriaConfig>
void invoke_hybrid_mc_raptor(d_query const& dq) {
  auto const& proc_stream = dq.mem_->context_.proc_stream_;
  auto const& transfer_stream = dq.mem_->context_.transfer_stream_;

  void* init_args[] = {(void*)&dq, (void*)dq.mem_->active_device_,  // NOLINT
                       (void*)&dq.tt_};  // NOLINT

  launch_kernel(mc_init_arrivals_kernel<CriteriaConfig>, init_args,
                dq.mem_->context_, proc_stream, dq.criteria_config_);
  cuda_sync_stream(proc_stream);

  fetch_arrivals_async(dq, 0, transfer_stream);

  for (int k = 1; k < max_raptor_round; ++k) {
    void* kernel_args[] = {(void*)dq.mem_->active_device_, (void*)&k,  // NOLINT
                           (void*)&dq.target_, (void*)&dq.tt_};  // NOLINT

    launch_kernel(mc_update_routes_kernel<CriteriaConfig>, kernel_args,
                  dq.mem_->context_, proc_stream, dq.criteria_config_);
    cuda_sync_stream(proc_stream);

    launch_kernel(mc_update_footpaths_kernel<CriteriaConfig>, kernel_args,
                  dq.mem_->context_, proc_stream, dq.criteria_config_);
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