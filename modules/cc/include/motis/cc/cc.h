#pragma once

#include "motis/module/module.h"

namespace motis::cc {

struct cc : public motis::module::module {
  cc() : module("Connection Checker", "cc") {}
  ~cc() override = default;

  cc(cc const&) = delete;
  cc& operator=(cc const&) = delete;

  cc(cc&&) = delete;
  cc& operator=(cc&&) = delete;

  void init(motis::module::registry&) override;

private:
  static motis::module::msg_ptr check_journey(motis::module::msg_ptr const&);
};

}  // namespace motis::cc
