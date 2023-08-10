#pragma once

#include <memory>
#include <vector>

#include "motis/paxmon/universe.h"

#include "motis/paxforecast/error.h"

namespace motis::paxforecast {

template <typename T>
struct universe_storage {
  universe_storage() { universe_created(0); }

  void universe_created(motis::paxmon::universe_id const id) {
    if (id >= storage_.size()) {
      storage_.resize(id + 1);
    }
    if (storage_[id] == nullptr) {
      storage_[id] = std::make_unique<T>();
    }
  }

  void universe_destroyed(motis::paxmon::universe_id const id) {
    if (id < storage_.size()) {
      storage_[id].reset(nullptr);
    }
  }

  T& get(motis::paxmon::universe_id const id) {
    if (id >= storage_.size()) {
      throw std::system_error{error::universe_not_found};
    }
    auto* m = storage_.at(id).get();
    if (m == nullptr) {
      throw std::system_error{error::universe_not_found};
    }
    return *m;
  }

  std::vector<std::unique_ptr<T>> storage_;
};

}  // namespace motis::paxforecast
