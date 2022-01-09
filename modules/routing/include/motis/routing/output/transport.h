#pragma once

#include "motis/core/schedule/connection.h"

namespace motis::routing::output::intermediate {

struct transport {
  transport() = default;
  transport(unsigned const from, unsigned const to,
            generic_light_connection con)
      : from_{from},
        to_{to},
        con_{std::move(con)},
        duration_{con.travel_time()},
        mumo_id_{0},
        mumo_price_{0},
        mumo_accessibility_{0} {}

  transport(unsigned const from, unsigned const to, duration_t const duration,
            int const mumo_id, unsigned const mumo_price,
            unsigned const mumo_accessibility)
      : from_(from),
        to_(to),
        con_(nullptr),
        duration_(duration),
        mumo_id_(mumo_id),
        mumo_price_(mumo_price),
        mumo_accessibility_(mumo_accessibility) {}

  bool is_walk() const { return con_ == nullptr; }

  unsigned from_, to_;
  generic_light_connection con_;
  duration_t duration_;
  int mumo_id_;
  unsigned mumo_price_;
  unsigned mumo_accessibility_;
};

}  // namespace motis::routing::output::intermediate
