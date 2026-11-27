#pragma once

#include <vector>

#include "opentelemetry/context/context.h"

#include "ctx/op_id.h"
#include "ctx/operation.h"

namespace motis {

struct ctx_data {
  void transition(ctx::transition, ctx::op_id, ctx::op_id) {}

  std::vector<opentelemetry::context::Context> otel_context_stack_;
};

}  // namespace motis