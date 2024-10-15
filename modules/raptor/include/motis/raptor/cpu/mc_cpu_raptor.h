#pragma once

#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/criteria/configs.h"

namespace motis::raptor {

template <typename CriteriaConfig>
void invoke_mc_cpu_raptor(raptor_query const& query, raptor_statistics&);

}  // namespace motis::raptor