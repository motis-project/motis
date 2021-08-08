#pragma once

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"
#include "motis/module/future.h"
#include "motis/module/message.h"

namespace motis::module {

inline std::vector<future> motis_publish_impl(msg_ptr const& msg,
                                              ctx::op_id id) {
  //  auto const op = ctx::current_op<ctx_data>();
  //  auto& data = op->data_;
  //  id.parent_index = op->id_.index;
  //  return data.dispatcher_->publish(msg, data, id);
  ctx_data d{ctx::access_t::READ, ctx_data::the_dispatcher_, nullptr};
  return ctx_data::the_dispatcher_->publish(msg, d, id);
}

#define motis_publish(msg) \
  ::motis::module::motis_publish_impl(msg, ctx::op_id(CTX_LOCATION))

}  // namespace motis::module
