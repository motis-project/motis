#pragma once

#include "motis/raptor/raptor_query.h"


namespace motis::raptor {

template<typename CriteriaConfig>
void invoke_mc_gpu_raptor(d_query const&);

}