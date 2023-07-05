#pragma once

#include <memory>

#include "motis/module/module.h"

namespace motis::valhalla {

struct valhalla : public motis::module::module {
  valhalla();
  ~valhalla() noexcept override;

  valhalla(valhalla const&) = delete;
  valhalla& operator=(valhalla const&) = delete;

  valhalla(valhalla&&) = delete;
  valhalla& operator=(valhalla&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;

  motis::module::msg_ptr one_to_many(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr table(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr via(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr ppr(motis::module::msg_ptr const&) const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::valhalla
