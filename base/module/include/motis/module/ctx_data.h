#pragma once

#include "ctx/access_data.h"
#include "ctx/access_scheduler.h"
#include "ctx/operation.h"

#include "motis/module/shared_data.h"

namespace motis::module {

struct dispatcher;

struct ctx_data : public ctx::access_data {
  ctx_data(dispatcher* d, shared_data* shared_data)
      : dispatcher_{d}, shared_data_{shared_data} {}

  void transition(ctx::transition, ctx::op_id const&, ctx::op_id const&) {}

  dispatcher* dispatcher_;
  shared_data* shared_data_;
};

inline ctx_data& current_data() { return ctx::current_op<ctx_data>()->data_; }

}  // namespace motis::module
