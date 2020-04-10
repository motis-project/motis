#pragma once

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"

namespace motis {

struct schedule;

namespace module {

inline schedule& get_schedule() {
  return *ctx::current_op<ctx_data>()->data_.dispatcher_->registry_.sched_;
}

}  // namespace module
}  // namespace motis
