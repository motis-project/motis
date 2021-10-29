namespace motis::raptor {

void cuda_sync_stream(cudaStream_t const& stream) {
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event,
                           cudaEventBlockingSync | cudaEventDisableTiming);
  cudaEventRecord(event, stream);
  cudaEventSynchronize(event);
  cudaEventDestroy(event);

  cc();
}

void fetch_arrivals_async(d_query& dq, raptor_round const round_k,
                          cudaStream_t s) {
  cudaMemcpyAsync((*dq.result_)[round_k], dq.d_arrivals_[round_k],
                  dq.stop_count_ * sizeof(time), cudaMemcpyDeviceToHost, s);
  cc();
}

void invoke_hybrid_raptor(d_query& dq) {
  auto const& device = *dq.device_;
  auto const& proc_stream = dq.proc_stream_;
  auto const& transfer_stream = dq.transfer_stream_;
  auto const mp = dq.mp_per_query_;

  void* init_args[] = {(void*)&dq};

  launch_kernel(init_arrivals_kernel, init_args, *dq.device_, proc_stream, mp);
  cuda_sync_stream(proc_stream);

  fetch_arrivals_async(dq, 0, transfer_stream);

  for (int k = 1; k < max_raptor_round; ++k) {
    void* kernel_args[] = {(void*)&dq, (void*)&k};

    launch_kernel(update_routes_kernel, kernel_args, device, proc_stream, mp);
    cuda_sync_stream(proc_stream);

    launch_kernel(update_footpaths_kernel, kernel_args, device, proc_stream,
                  mp);
    cuda_sync_stream(proc_stream);

    cudaMemcpyAsync(dq.any_station_marked_h_, dq.any_station_marked_d_,
                    sizeof(bool), cudaMemcpyDeviceToHost, transfer_stream);
    cuda_sync_stream(transfer_stream);

    if (!dq.any_station_marked_h_) {
      break;
    }

    fetch_arrivals_async(dq, k, transfer_stream);
  }

  //  cuda_sync_stream(proc_stream);
  cuda_sync_stream(transfer_stream);
}

}  // namespace motis::raptor