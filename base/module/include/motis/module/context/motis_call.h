#pragma once

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"
#include "motis/module/future.h"
#include "motis/module/message.h"

namespace motis::module {

inline future motis_call_impl(msg_ptr const& msg, ctx::op_id id) {
  if (ctx_data::direct_mode_dispatcher_ != nullptr) {
    ctx_data d{ctx::access_t::READ, ctx_data::direct_mode_dispatcher_, nullptr};
    return ctx_data::direct_mode_dispatcher_->req(msg, d, id);
  } else {
    auto const op = ctx::current_op<ctx_data>();
    id.parent_index = op->id_.index;
    return ctx_data::direct_mode_dispatcher_->req(msg, op->data_, id);
  }
}

#define motis_call(msg) \
  motis::module::motis_call_impl(msg, ctx::op_id(CTX_LOCATION))

}  // namespace motis::module
