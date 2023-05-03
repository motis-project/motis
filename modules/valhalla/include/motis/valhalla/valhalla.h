#pragma once

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

private:
  motis::module::msg_ptr route(motis::module::msg_ptr const&);
};

}  // namespace motis::valhalla
