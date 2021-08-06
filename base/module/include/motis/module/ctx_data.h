#pragma once

#include "ctx/access_scheduler.h"
#include "ctx/operation.h"

#include "motis/module/shared_data.h"

namespace motis {

namespace module {

struct dispatcher;

struct ctx_data {
  ctx_data(ctx::access_t access, dispatcher* d, shared_data* shared_data)
      : access_{access}, dispatcher_{d}, shared_data_{shared_data} {}

  void transition(ctx::transition, ctx::op_id const&, ctx::op_id const&) {}

  ctx::access_t access_;
  dispatcher* dispatcher_;
  shared_data* shared_data_;

  static dispatcher* the_dispatcher_;
};

inline ctx_data& current_data() { return ctx::current_op<ctx_data>()->data_; }

}  // namespace module
}  // namespace motis
