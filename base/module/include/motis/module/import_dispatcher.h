#pragma once

#include <functional>
#include <vector>

#include "motis/module/message.h"

namespace motis::module {

struct import_dispatcher {
  using importer_fn = std::function<void(msg_ptr)>;

  void subscribe(importer_fn&& i) { importers_.emplace_back(std::move(i)); }

  void publish(msg_ptr const& m) { publish_queue_.emplace_back(m); }

  void run() {
    while (!publish_queue_.empty()) {
      auto const m = publish_queue_.front();
      publish_queue_.erase(begin(publish_queue_));
      for (auto const& i : importers_) {
        i(m);
      }
    }
  }

  std::vector<importer_fn> importers_;
  std::vector<msg_ptr> publish_queue_;
};

}  // namespace motis::module
