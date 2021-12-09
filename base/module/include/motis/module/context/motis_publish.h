#pragma once

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"
#include "motis/module/future.h"
#include "motis/module/message.h"

namespace motis::module {

inline std::vector<future> motis_publish_impl(msg_ptr const& msg,
                                              ctx::op_id id) {
  if (dispatcher::direct_mode_dispatcher_ != nullptr) {
    ctx_data d{dispatcher::direct_mode_dispatcher_};
    return dispatcher::direct_mode_dispatcher_->publish(msg, d, id);
  } else {
    auto const op = ctx::current_op<ctx_data>();
    auto& data = op->data_;
    id.parent_index = op->id_.index;
    return data.dispatcher_->publish(msg, data, id);
  }
}

#define motis_publish(msg) \
  ::motis::module::motis_publish_impl(msg, ctx::op_id(CTX_LOCATION))

}  // namespace motis::module
