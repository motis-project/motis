#pragma once

#include "ctx/ctx.h"
#include "ctx/parallel_for.h"

#include "utl/parallel_for.h"

#include "motis/module/dispatcher.h"

namespace motis::module {

template <typename Vec, typename Fn>
void motis_parallel_for_impl(Vec&& vec, Fn&& fn, ctx::op_id const id) {
  ctx::parallel_for<motis::module::ctx_data>(std::forward<Vec>(vec),
                                             std::forward<Fn>(fn), id);
}

}  // namespace motis::module

#define motis_parallel_for(vec, fn) \
  ::motis::module::motis_parallel_for_impl(vec, fn, ctx::op_id(CTX_LOCATION))
