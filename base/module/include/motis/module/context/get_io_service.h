#pragma once

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"

namespace motis::module {

inline boost::asio::io_service& get_io_service() {
  if (dispatcher::direct_mode_dispatcher_ != nullptr) {
    return dispatcher::direct_mode_dispatcher_->runner_.ios();
  } else {
    return ctx::current_op<ctx_data>()->data_.dispatcher_->runner_.ios();
  }
}

}  // namespace motis::module
