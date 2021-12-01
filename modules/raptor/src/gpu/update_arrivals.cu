#include "motis/raptor/gpu/update_arrivals.cuh"


namespace motis::raptor {

__device__ bool update_arrival(time* const base, stop_id const s_id,
                               time const val) {

#if __CUDA_ARCH__ >= 700

  auto old_value = base[s_id];
  time assumed{};

  do {
    if (old_value <= val) {
      return false;
    }

    assumed = old_value;

    old_value = atomicCAS(&base[s_id], assumed, val);
  } while (assumed != old_value);

  return true;

#else

  // we have a 16-bit time value array, but only 32-bit atomic operations
  // therefore every two 16-bit time values are read as one 32-bit time value
  // then they are the corresponding part is updated and stored if a better
  // time value was found while the remaining 16 bit value part remains
  // unchanged

  time* const arr_address = &base[s_id];
  unsigned int* base_address = (unsigned int*)((size_t)arr_address & ~2);
  unsigned int old_value, assumed, new_value, compare_val;

  old_value = *base_address;

  do {
    assumed = old_value;

    if ((size_t)arr_address & 2) {
      compare_val = (0x0000FFFF & assumed) ^ (((unsigned int)val) << 16);
    } else {
      compare_val = (0xFFFF0000 & assumed) ^ (unsigned int)val;
    }

    new_value = __vminu2(old_value, compare_val);

    if (new_value == old_value) {
      return false;
    }

    old_value = atomicCAS(base_address, assumed, new_value);
  } while (assumed != old_value);

  return true;

#endif
}

}