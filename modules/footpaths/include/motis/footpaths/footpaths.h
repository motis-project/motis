#pragma once

#include "motis/module/module.h"

namespace motis::footpaths {

struct footpaths : public motis::module::module {
  footpaths();
  ~footpaths() override;

  footpaths(footpaths const&) = delete;
  footpaths& operator=(footpaths const&) = delete;

  footpaths(footpaths&&) = delete;
  footpaths& operator=(footpaths&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; };

private:
  int max_walk_duration_{15};
  bool wheelchair_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
};

}  // namespace motis::footpaths
