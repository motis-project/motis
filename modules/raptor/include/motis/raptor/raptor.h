#pragma once

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/cuda_util.h"
#include "motis/raptor/gpu/devices.h"
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/memory_store.h"
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
  template <class Query>
  Query get_query(motis::routing::RoutingRequest const*, schedule const&);

  template <typename Query, typename RaptorFun>
  motis::module::msg_ptr route_generic(motis::module::msg_ptr const&,
                                       RaptorFun const&);

  std::unique_ptr<raptor_schedule> raptor_sched_;
  std::unique_ptr<raptor_timetable> timetable_;

#if defined(MOTIS_CUDA)
  std::unique_ptr<host_gpu_timetable> h_gtt_;
  std::unique_ptr<device_gpu_timetable> d_gtt_;

  devices devices_;

  int32_t mp_per_query_;

  memory_store mem_store_;

#endif
};

}  // namespace motis::raptor