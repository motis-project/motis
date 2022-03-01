#pragma once

#include <vector>

#include "motis/module/module.h"

namespace motis::isochrone {

struct memory;

struct isochrone : public motis::module::module {
  isochrone();
  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr list_stations(const motis::module::msg_ptr& msg);
};

}  // namespace motis::isochrone
