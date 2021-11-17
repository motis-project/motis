#include "motis/raptor/gpu/mc_gpu_raptor.cuh"

#include "motis/raptor/criteria/configs.h"


namespace motis::raptor {



template<typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const& dq) {
  //TODO implement
}

RAPTOR_CRITERIA_CONFIGS_WO_DEFAULT(MAKE_MC_GPU_RAPTOR_TEMPLATE_INSTANCE, )

}