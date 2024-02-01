#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "motis/core/common/unixtime.h"

#include "motis/paxmon/loaded_files.h"
#include "motis/paxmon/multiverse.h"

#include "motis/paxmon/eval/forecast/pax_check_data.h"

namespace motis::paxmon {

struct paxmon_data {
  explicit paxmon_data(motis::module::module& mod)
      : multiverse_{std::make_shared<multiverse>(mod)} {}

  std::shared_ptr<multiverse> multiverse_;

  std::chrono::seconds max_universe_ttl_{std::chrono::minutes{30}};
  bool allow_infinite_universe_ttl_{true};

  std::vector<loaded_journey_file> loaded_journey_files_;
  std::vector<loaded_capacity_file> loaded_capacity_files_;

  unixtime motis_start_time_{};

  eval::forecast::pax_check_data pax_check_data_;
};

}  // namespace motis::paxmon
