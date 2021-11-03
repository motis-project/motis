#pragma once

#include "motis/raptor/raptor_query.h"

namespace motis::raptor {

void invoke_gpu_raptor(d_query const&);
void invoke_hybrid_raptor(d_query const&);

}  // namespace motis::raptor