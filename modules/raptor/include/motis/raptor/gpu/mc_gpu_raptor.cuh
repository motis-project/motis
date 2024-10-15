#pragma once

#include "motis/raptor/raptor_query.h"

namespace motis::raptor {

#define GENERATE_LAUNCH_CONFIG_FUNC_DECL(VAL, ACCESSOR)           \
  template <>                                                     \
  std::pair<dim3, dim3> get_mc_gpu_raptor_launch_parameters<VAL>( \
      device_id const device_id, int32_t const concurrency_per_device);

template <typename CriteriaConfig>
std::pair<dim3, dim3> get_mc_gpu_raptor_launch_parameters(
    device_id const device_id, int32_t const concurrency_per_device) {
  throw std::system_error{access::error::not_implemented};
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(GENERATE_LAUNCH_CONFIG_FUNC_DECL,
                                   raptor_criteria_config)

template <typename CriteriaConfig>
__device__ void mc_update_routes_dev(device_memory const&,
                                     raptor_round,
                                     stop_id,
                                     device_gpu_timetable const&);

template <typename CriteriaConfig>
__device__ void mc_update_footpaths_dev(device_memory const&,
                                        raptor_round, stop_id,
                                        device_gpu_timetable const&);

template <typename CriteriaConfig>
__device__ void mc_init_arrivals_dev(base_query const&, device_memory const&,
                                     device_gpu_timetable const&);

template <typename CriteriaConfig>
void invoke_hybrid_mc_raptor(d_query const&);

template <typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const&);

}  // namespace motis::raptor