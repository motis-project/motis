#pragma once

#include <memory>

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

#if defined(MOTIS_CUDA)
#include "motis/raptor/gpu/gpu_timetable.cuh"
#include "motis/raptor/gpu/memory_store.h"
#endif

namespace motis::raptor {

struct config {
#if defined(MOTIS_CUDA)
  int32_t queries_per_device_{1};
#endif
};

struct raptor : public motis::module::module {
  raptor();
  ~raptor() override;

  raptor(raptor const&) = delete;
  raptor& operator=(raptor const&) = delete;

  raptor(raptor&&) = delete;
  raptor& operator=(raptor&&) = delete;

  void init(motis::module::registry&) override;

private:
  struct impl;
  std::unique_ptr<impl> impl_;

  config config_;
};

}  // namespace motis::raptor