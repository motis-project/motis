#pragma once

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/gpu/memory_store.h"
#include "motis/raptor/raptor_query.h"
#endif

namespace motis::raptor {

struct raptor : public motis::module::module {
  raptor();
  ~raptor() override;

  raptor(raptor const&) = delete;
  raptor& operator=(raptor const&) = delete;

  raptor(raptor&&) = delete;
  raptor& operator=(raptor&&) = delete;

  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr route_cpu(motis::module::msg_ptr const& msg);
  motis::module::msg_ptr route_gpu(motis::module::msg_ptr const& msg);

  std::unique_ptr<raptor_schedule> raptor_sched_;
  std::unique_ptr<raptor_timetable> timetable_;

#if defined(MOTIS_CUDA)
  std::unique_ptr<host_gpu_timetable> h_gtt_;
  std::unique_ptr<device_gpu_timetable> d_gtt_;

  int32_t queries_per_device_{1};

  memory_store mem_store_;
#endif
};

}  // namespace motis::raptor