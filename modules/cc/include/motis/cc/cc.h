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
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override;

private:
  motis::module::msg_ptr check_journey(motis::module::msg_ptr const&);

  bool import_successful_{false};
};

}  // namespace motis::cc
