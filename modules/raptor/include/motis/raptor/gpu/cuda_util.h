#pragma once

#include <cstdio>

#include "cuda_runtime.h"

namespace motis::raptor {

#if __CUDA_ARCH__ <= 720
#define MAX_BLOCKS_PER_SM 32
#define MAX_THREADS_PER_SM 2048
#elif __CUDA_ARCH__ == 750
#define MAX_BLOCKS_PER_SM 16
#define MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ == 800
#define MAX_BLOCKS_PER_SM 32
#define MAX_THREADS_PER_SM 2048
#elif __CUDA_ARCH__ == 850
#define MAX_BLOCKS_PER_SM 16
#define MAX_THREADS_PER_SM 1536
#endif

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