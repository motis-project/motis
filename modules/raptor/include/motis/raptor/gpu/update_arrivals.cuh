#pragma once

#include "motis/raptor/gpu/gpu_timetable.cuh"

namespace motis::raptor {

__device__ bool update_arrival(time* const base, stop_id const s_id,
                               time const val);

}