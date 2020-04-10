#pragma once

#include <system_error>

#include "motis/module/message.h"

namespace motis::module {

using callback = std::function<void(msg_ptr, std::error_code)>;

struct receiver {
  receiver() = default;
  receiver(receiver const&) = default;
  receiver(receiver&&) = default;
  receiver& operator=(receiver const&) = default;
  receiver& operator=(receiver&&) = default;
  virtual ~receiver() = default;
  virtual void on_msg(msg_ptr const&, callback const&) = 0;
};

}  // namespace motis::module
