#pragma once

#include <vector>

#include "ctx/access_data.h"
#include "ctx/access_scheduler.h"
#include "ctx/operation.h"

#include "opentelemetry/context/context.h"

namespace motis::module {

struct dispatcher;

struct ctx_data : public ctx::access_data {
  explicit ctx_data(dispatcher* d) : dispatcher_{d} {}

  void transition(ctx::transition, ctx::op_id const&, ctx::op_id const&) {}

  dispatcher* dispatcher_;
  std::vector<opentelemetry::context::Context> otel_context_stack_;
};

inline ctx_data& current_data() { return ctx::current_op<ctx_data>()->data_; }

}  // namespace motis::module
