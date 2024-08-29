#pragma once

#include <memory>

#include "motis/module/module.h"

namespace motis::metrics {

struct metrics : public motis::module::module {
  metrics();
  ~metrics() noexcept override;

  metrics(metrics const&) = delete;
  metrics& operator=(metrics const&) = delete;

  metrics(metrics&&) = delete;
  metrics& operator=(metrics&&) = delete;

  void init(motis::module::registry&) override;

  motis::module::msg_ptr request(motis::module::msg_ptr const&) const;
};

}  // namespace motis::metrics
