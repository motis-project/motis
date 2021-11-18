#pragma once

#include "motis/raptor/gpu/gpu_timetable.cuh"

namespace motis::raptor {

__device__ void mark(unsigned int* store, unsigned int const idx);

__device__ bool marked(unsigned int const* const store, unsigned int idx);

__device__ void reset_store(unsigned int* store, int const store_size);

__device__ void convert_station_to_route_marks(unsigned int* station_marks,
                                               unsigned int* route_marks,
                                               bool* any_station_marked,
                                               device_gpu_timetable const& tt);
}