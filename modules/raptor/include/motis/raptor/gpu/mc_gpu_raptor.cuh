#pragma once

#include "motis/raptor/raptor_query.h"

namespace motis::raptor {

#define GENERATE_LAUNCH_CONFIG_FUNC_DECL(VAL, ACCESSOR) \
  template<>                                          \
  std::tuple<int, int> get_mc_gpu_launch_config<VAL>();

template<typename CriteriaConfig>
std::tuple<int, int> get_mc_gpu_launch_config() {
  throw std::system_error{access::error::not_implemented};
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(GENERATE_LAUNCH_CONFIG_FUNC_DECL,
                                   raptor_criteria_config)

template <typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const&);

}  // namespace motis::raptor