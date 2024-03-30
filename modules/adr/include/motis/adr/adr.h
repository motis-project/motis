#pragma once

#include "motis/module/module.h"

namespace motis::adr {

struct adr : public motis::module::module {
  adr();
  ~adr() override;

  adr(adr const&) = delete;
  adr& operator=(adr const&) = delete;

  adr(adr&&) = delete;
  adr& operator=(adr&&) = delete;

  void import(motis::module::import_dispatcher&) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

private:
  motis::module::msg_ptr guess(motis::module::msg_ptr const&);

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
};

}  // namespace motis::adr
