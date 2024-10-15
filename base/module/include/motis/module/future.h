#pragma once

#include <memory>

#include "ctx/future.h"

#include "motis/module/ctx_data.h"
#include "motis/module/message.h"

namespace motis::module {

using future = std::shared_ptr<ctx::future<ctx_data, msg_ptr>>;

inline future make_future(ctx::op_id const& parent) {
  return std::make_shared<ctx::future<ctx_data, msg_ptr>>(parent);
}

}  // namespace motis::module
