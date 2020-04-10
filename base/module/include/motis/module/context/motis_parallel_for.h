#pragma once

#include "ctx/ctx.h"
#include "ctx/parallel_for.h"

#include "motis/module/ctx_data.h"

#define motis_parallel_for(vec, fn) \
  ctx::parallel_for<motis::module::ctx_data>(vec, fn, ctx::op_id(CTX_LOCATION))
