#pragma once

#include <memory>
#include <vector>

#include "motis/paxmon/universe.h"

#include "motis/paxforecast/error.h"
#include "motis/paxforecast/measures/measures.h"

namespace motis::paxforecast::measures {

struct storage {
  storage() { universe_created(0); }

  void universe_created(motis::paxmon::universe_id const id) {
    if (id >= measures_for_universe_.size()) {
      measures_for_universe_.resize(id + 1);
    }
    if (measures_for_universe_[id] == nullptr) {
      measures_for_universe_[id] = std::make_unique<measure_collection>();
    }
  }

  void universe_destroyed(motis::paxmon::universe_id const id) {
    if (id < measures_for_universe_.size()) {
      measures_for_universe_[id].reset(nullptr);
    }
  }

  measure_collection& get(motis::paxmon::universe_id const id) {
    if (id >= measures_for_universe_.size()) {
      throw std::system_error{error::universe_not_found};
    }
    auto* m = measures_for_universe_.at(id).get();
    if (m == nullptr) {
      throw std::system_error{error::universe_not_found};
    }
    return *m;
  }

  std::vector<std::unique_ptr<measure_collection>> measures_for_universe_;
};

}  // namespace motis::paxforecast::measures
