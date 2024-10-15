#pragma once

#include <variant>

#include "ctx/access_scheduler.h"
#include "ctx/res_id_t.h"

#include "utl/overloaded.h"

#include "motis/module/ctx_data.h"

namespace motis::module {

struct locked_resources {
  template <typename T>
  T& get(ctx::res_id_t const res_id) {
    return std::visit(
        utl::overloaded{[&](ctx::access_scheduler<ctx_data>::mutex& m) -> T& {
                          return m.get<T>(res_id);
                        },
                        [&](ctx::access_scheduler<ctx_data>* s) -> T& {
                          return s->get<T>(res_id);
                        }},
        mutex_or_scheduler_);
  }

  std::variant<ctx::access_scheduler<ctx_data>::mutex,
               ctx::access_scheduler<ctx_data>*>
      mutex_or_scheduler_;
};

}  // namespace motis::module
