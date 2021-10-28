#include "motis/raptor/gpu/gpu_raptor.cuh"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/raptor_query.h"

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

inline auto get_kernel_launch_config_from_props(cudaDeviceProp const& gp) {
  kernel_launch_config klc;

  unsigned int block_dim_x = 32;  // must always be 32!
  unsigned int block_dim_y = 32;  // range [1, ..., 32]
  unsigned int block_size = block_dim_x * block_dim_y;

  unsigned int min_blocks_per_sm = gp.maxThreadsPerMultiProcessor / block_size;
  unsigned int num_blocks = gp.multiProcessorCount * min_blocks_per_sm;

  klc.threads_per_block_ = dim3(block_dim_x, block_dim_y, 1);
  klc.grid_ = dim3(num_blocks, 1, 1);

  return klc;
}

template <typename Kernel>
void inline launch_kernel(Kernel kernel, void** args, device const& device,
                          cudaStream_t stream) {
  cudaSetDevice(device.id_);
  auto const& config = get_kernel_launch_config_from_props(device.props_);

  // TOOD(julian) with this enable the max throughput
  // device.props_.concurrentKernels

  cudaLaunchCooperativeKernel((void*)kernel, config.grid_,
                              config.threads_per_block_, args, 0, stream);
  cc();
}

}  // namespace motis::raptor

#include "gpu_raptor.cu"
#include "gpu_timetable.cu"
#include "hybrid_raptor.cu"
