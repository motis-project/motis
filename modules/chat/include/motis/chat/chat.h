#pragma once

#include <mutex>
#include <vector>

#include "motis/module/module.h"

namespace motis::chat {

struct chat : public motis::module::module {
  chat() : module("Chat", "chat") {}
  ~chat() override = default;

  chat(chat const&) = delete;
  chat& operator=(chat const&) = delete;

  chat(chat&&) = delete;
  chat& operator=(chat&&) = delete;

  void init(motis::module::registry&) override;

  std::mutex clients_lock_;
  std::vector<motis::module::client_hdl> clients_;
};

}  // namespace motis::chat
