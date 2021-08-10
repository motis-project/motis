#pragma once

#include "ctx/ctx.h"
#include "ctx/parallel_for.h"

#include "utl/parallel_for.h"

#include "motis/module/dispatcher.h"

namespace motis::module {

template <typename Vec, typename Fn>
void motis_parallel_for_impl(Vec&& vec, Fn&& fn, ctx::op_id const id) {
  if (dispatcher::direct_mode_dispatcher_ != nullptr) {
    ctx::parallel_for<motis::module::ctx_data>(std::forward<Vec>(vec),
                                               std::forward<Fn>(fn), id);
  } else {
    utl::parallel_for(std::forward<Vec>(vec), std::forward<Fn>(fn));
  }
}

}  // namespace motis::module

#define motis_parallel_for(vec, fn) \
  motis_parallel_for_impl(vec, fn, ctx::op_id(CTX_LOCATION))
