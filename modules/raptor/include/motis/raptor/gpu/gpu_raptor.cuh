#pragma once

#include "motis/raptor/raptor_query.h"

namespace motis::raptor {

void invoke_gpu_raptor(d_query&);
void invoke_hybrid_raptor(d_query&);

}  // namespace motis::raptor