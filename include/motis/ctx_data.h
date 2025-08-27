#pragma once

#include "ctx/op_id.h"
#include "ctx/operation.h"

namespace motis {

struct ctx_data {
  void transition(ctx::transition, ctx::op_id, ctx::op_id) {}
};

}  // namespace motis