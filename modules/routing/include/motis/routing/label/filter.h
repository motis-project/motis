#pragma once

#include "motis/core/schedule/time.h"

namespace motis::routing {

template <typename... Filters>
struct filter {
  template <typename Label>
  static bool is_filtered(Label const& l, duration const fastest_direct) {
    return (Filters::is_filtered(l, fastest_direct) || ...);
  }
};

}  // namespace motis::routing
