namespace motis::raptor {

void invoke_hybrid_raptor(d_query& dq) {
  auto const& proc_stream = dq.context_.proc_stream_;
  auto const& transfer_stream = dq.context_.transfer_stream_;

  void* init_args[] = {(void*)&dq};

  launch_kernel(init_arrivals_kernel, init_args, dq.context_, proc_stream);
  cuda_sync_stream(proc_stream);

  fetch_arrivals_async(dq, 0, transfer_stream);

  for (int k = 1; k < max_raptor_round; ++k) {
    void* kernel_args[] = {(void*)&dq, (void*)&k};

    launch_kernel(update_routes_kernel, kernel_args, dq.context_, proc_stream);
    cuda_sync_stream(proc_stream);

    launch_kernel(update_footpaths_kernel, kernel_args, dq.context_,
                  proc_stream);
    cuda_sync_stream(proc_stream);

    cudaMemcpyAsync(dq.host_.any_station_marked_,
                    dq.device_.any_station_marked_, sizeof(bool),
                    cudaMemcpyDeviceToHost, transfer_stream);
    cuda_sync_stream(transfer_stream);

    if (!dq.host_.any_station_marked_) {
      break;
    }

    fetch_arrivals_async(dq, k, transfer_stream);
  }

  //  cuda_sync_stream(proc_stream);
  cuda_sync_stream(transfer_stream);
}

}  // namespace motis::raptor