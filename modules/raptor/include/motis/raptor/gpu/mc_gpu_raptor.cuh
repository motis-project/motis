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
void invoke_mc_gpu_raptor(d_query const&);

}  // namespace motis::raptor