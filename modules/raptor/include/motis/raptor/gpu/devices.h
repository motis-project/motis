#pragma once

#include "cuda_runtime.h"

namespace motis::raptor {

struct device {
  int id_;
  cudaDeviceProp props_;
};

using devices = std::vector<device>;

inline devices get_devices() {
  devices ds;

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  cc();

  for (int device_id = 0; device_id < device_count; ++device_id) {
    cudaSetDevice(device_id);
    cc();

    device d{};
    d.id_ = device_id;

    cudaGetDeviceProperties(&d.props_, device_id);
    cc();

    ds.push_back(d);
  }

  return ds;
}

}  // namespace motis::raptor