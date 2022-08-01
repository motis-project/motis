#pragma once

#include <chrono>
#include <memory>

#include "motis/paxmon/capacity_maps.h"
#include "motis/paxmon/multiverse.h"

namespace motis::paxmon {

struct paxmon_data {
  explicit paxmon_data(motis::module::module& mod)
      : multiverse_{std::make_shared<multiverse>(mod)} {}

  capacity_maps capacity_maps_;
  std::shared_ptr<multiverse> multiverse_;

  std::chrono::seconds max_universe_ttl_{std::chrono::minutes{30}};
  bool allow_infinite_universe_ttl_{true};
};

}  // namespace motis::paxmon
