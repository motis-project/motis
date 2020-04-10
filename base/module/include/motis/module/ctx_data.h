#pragma once

#include "ctx/access_scheduler.h"
#include "ctx/operation.h"

namespace motis {

struct schedule;

namespace module {

struct dispatcher;

struct ctx_data {
  ctx_data(ctx::access_t access, dispatcher* d, schedule* sched)
      : access_{access}, dispatcher_{d}, sched_{sched} {}

  void transition(ctx::transition, ctx::op_id const&, ctx::op_id const&) {}

  ctx::access_t access_;
  dispatcher* dispatcher_;
  schedule* sched_;
};

inline ctx_data& current_data() { return ctx::current_op<ctx_data>()->data_; }

}  // namespace module
}  // namespace motis
