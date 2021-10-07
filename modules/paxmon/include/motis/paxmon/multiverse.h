#pragma once

#include <mutex>
#include <optional>
#include <stdexcept>

#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct multiverse {
  multiverse() { universes_.emplace_back(mcd::make_unique<universe>()); }

  universe& primary() {
    std::lock_guard lock{mutex_};
    return *universes_.front();
  }

  universe& get(universe_id const id) {
    std::lock_guard lock{mutex_};
    auto* ptr = universes_.at(id).get();
    if (ptr != nullptr) {
      return *ptr;
    } else {
      throw std::runtime_error{"requested paxmon universe already destroyed"};
    }
  }

  std::optional<universe*> try_get(universe_id const id) {
    std::lock_guard lock{mutex_};
    if (id < universes_.size()) {
      if (auto* ptr = universes_[id].get(); ptr != nullptr) {
        return {ptr};
      }
    }
    return {};
  }

  universe& fork(universe_id const base_id) {
    std::lock_guard lock{mutex_};
    auto const& base_uv = get(base_id);
    auto const new_id = universes_.size();
    auto& new_uv = universes_.emplace_back(mcd::make_unique<universe>(base_uv));
    new_uv->id_ = new_id;
    return *new_uv;
  }

  bool destroy(universe_id const id) {
    std::lock_guard lock{mutex_};
    if (id > 0 && id < universes_.size()) {
      if (auto& ptr = universes_[id]; ptr != nullptr) {
        ptr.reset();
        return true;
      }
    }
    return false;
  }

  mcd::vector<mcd::unique_ptr<universe>> universes_;
  std::mutex mutex_;
};

}  // namespace motis::paxmon
