#pragma once

#include "motis/core/journey/journey.h"

namespace motis::revise {

struct extern_interchange {
  extern_interchange(journey::stop first_stop, int const first_stop_idx,
                     journey::stop second_stop, int const second_stop_idx)
      : first_stop_{std::move(first_stop)},
        first_stop_idx_{first_stop_idx},
        second_stop_{std::move(second_stop)},
        second_stop_idx_{second_stop_idx} {}

  extern_interchange(journey::stop first_stop, int const first_stop_idx)
      : first_stop_{std::move(first_stop)}, first_stop_idx_{first_stop_idx} {}

  journey::stop first_stop_;
  int first_stop_idx_ = -1;
  journey::stop second_stop_;
  int second_stop_idx_ = -1;
};

}  // namespace motis::revise
