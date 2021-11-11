#pragma once

#include <cstdio>

#include "cuda_runtime.h"

namespace motis::raptor {

#define cucheck_dev(call)                                    \
  {                                                          \
    cudaError_t cucheck_err = (call);                        \
    if (cucheck_err != cudaSuccess) {                        \
      const char* err_str = cudaGetErrorString(cucheck_err); \
      printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);  \
    }                                                        \
  }

#define cuda_check() \
  { cucheck_dev(cudaGetLastError()); }

inline void cuda_sync_stream(cudaStream_t const& stream) {
  cudaEvent_t event{};
  cudaEventCreateWithFlags(&event,
                           cudaEventBlockingSync | cudaEventDisableTiming);
  cudaEventRecord(event, stream);
  cudaEventSynchronize(event);
  cudaEventDestroy(event);

  cuda_check();
}

}  // namespace motis::raptor