#pragma once

#include <memory>

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/mcraptor/raptor_statistics.h"
#include "motis/mcraptor/raptor_timetable.h"

namespace motis::mcraptor {

struct config {
#if defined(MOTIS_CUDA)
  int32_t queries_per_device_{1};
#endif
};

struct mcraptor : public motis::module::module {
  mcraptor();
  ~mcraptor() override;

  mcraptor(mcraptor const&) = delete;
  mcraptor& operator=(mcraptor const&) = delete;

  mcraptor(mcraptor&&) = delete;
  mcraptor& operator=(mcraptor&&) = delete;

  void init(motis::module::registry&) override;

private:
  struct impl;
  std::unique_ptr<impl> impl_;

  config config_;
};

}  // namespace motis::mcraptor