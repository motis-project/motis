#pragma once

#include <memory>

#include "motis/module/module.h"

namespace motis::osr {

struct osr : public motis::module::module {
  osr();
  ~osr() noexcept override;

  osr(osr const&) = delete;
  osr& operator=(osr const&) = delete;

  osr(osr&&) = delete;
  osr& operator=(osr&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;

  motis::module::msg_ptr one_to_many(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr table(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr via(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr ppr(motis::module::msg_ptr const&) const;

  bool import_successful() const override { return import_successful_; }

  bool import_successful_{false};
  bool lock_{true};

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::osr
