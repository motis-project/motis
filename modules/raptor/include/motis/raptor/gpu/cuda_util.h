#pragma once

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

#define cc() \
  { cucheck_dev(cudaGetLastError()); }

template <typename T>
inline void cuda_free(T* ptr) {
  cudaFree(ptr);
  cc();
}

template <typename T>
inline void cuda_malloc_set(T** ptr, size_t const bytes, char const value) {
  cudaMalloc(ptr, bytes);
  cc();
  cudaMemset(*ptr, value, bytes);
  cc();
}

}  // namespace motis::raptor