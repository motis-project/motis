#pragma once

#include "motis/config.h"
#include "motis/import/task.h"

namespace motis {

struct adr_import : public task {
  void load() override;
  void unload() override;
  void run() override;
  bool is_enabled() const override;
};

}  // namespace motis
