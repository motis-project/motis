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

struct kernel_launch_config {
  dim3 threads_per_block_;
  dim3 grid_;
};

inline auto get_kernel_launch_config_from_props(cudaDeviceProp const& gp,
                                                int32_t const mp_per_query) {
  kernel_launch_config klc;

  int32_t block_dim_x = 32;  // must always be 32!
  int32_t block_dim_y = 32;  // range [1, ..., 32]
  int32_t block_size = block_dim_x * block_dim_y;

  int32_t max_blocks_per_sm = gp.maxThreadsPerMultiProcessor / block_size;
  int32_t mp_count = std::min(mp_per_query, gp.multiProcessorCount);
  int32_t num_blocks = mp_count * max_blocks_per_sm;

  // TOOD(julian) with this enable the max throughput
  // device.props_.concurrentKernels

  klc.threads_per_block_ = dim3(block_dim_x, block_dim_y, 1);
  klc.grid_ = dim3(num_blocks, 1, 1);

  return klc;
}

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args, device const& device,
                          cudaStream_t stream, int32_t mp_per_query) {
  cudaSetDevice(device.id_);

  auto config =
      get_kernel_launch_config_from_props(device.props_, mp_per_query);

  cudaLaunchCooperativeKernel((void*)kernel, config.grid_,
                              config.threads_per_block_, args, 0, stream);
  cc();
}

}  // namespace motis::raptor

#include "gpu_raptor.cu"
#include "gpu_timetable.cu"
#include "hybrid_raptor.cu"
