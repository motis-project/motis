#pragma once

#include <functional>
#include <string_view>

#include "motis/module/message.h"

namespace motis::module {

struct client {
  client() = default;
  client(client const&) = delete;
  client(client const&&) = delete;
  client const& operator=(client const&) = delete;
  client const& operator=(client const&&) = delete;

  virtual ~client() = default;
  virtual void set_on_msg_cb(std::function<void(msg_ptr const&)>&&) = 0;
  virtual void set_on_close_cb(std::function<void()>&&) = 0;
  virtual void send(msg_ptr const&) = 0;
};

using client_hdl = std::weak_ptr<client>;

}  // namespace motis::module
