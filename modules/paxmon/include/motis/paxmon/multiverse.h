#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct multiverse {
  multiverse() { universes_.emplace_back(std::make_shared<universe>()); }

  universe& primary() {
    std::lock_guard lock{mutex_};
    return *universes_.front().get();
  }

  std::shared_ptr<universe> get(universe_id const id) {
    std::lock_guard lock{mutex_};
    auto ptr = universes_.at(id);
    if (ptr != nullptr) {
      return ptr;
    } else {
      throw std::runtime_error{"requested paxmon universe already destroyed"};
    }
  }

  std::optional<std::shared_ptr<universe>> try_get(universe_id const id) {
    std::lock_guard lock{mutex_};
    if (id < universes_.size()) {
      if (auto ptr = universes_[id]; ptr != nullptr) {
        return {ptr};
      }
    }
    return {};
  }

  std::shared_ptr<universe> fork(universe_id const base_id) {
    std::lock_guard lock{mutex_};
    auto const base_uv = get(base_id);
    auto const new_id = universes_.size();
    auto new_uv =
        universes_.emplace_back(std::make_shared<universe>(*base_uv.get()));
    new_uv->id_ = new_id;
    return new_uv;
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

  mcd::vector<std::shared_ptr<universe>> universes_;
  std::recursive_mutex mutex_;
};

}  // namespace motis::paxmon
