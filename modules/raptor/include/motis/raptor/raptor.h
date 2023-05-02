#pragma once

#include <memory>

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

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
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override;

private:
  struct impl;
  std::unique_ptr<impl> impl_;

  config config_;
  bool import_successful_{false};
};

}  // namespace motis::raptor