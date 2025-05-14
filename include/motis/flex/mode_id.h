#pragma once

#include "utl/verify.h"

#include "nigiri/types.h"

namespace motis::flex {

struct mode_id {
  mode_id(nigiri::flex_transport_idx_t const t,
          nigiri::stop_idx_t const stop_idx)
      : stop_idx_{stop_idx}, transport_{t} {
    utl::verify(t < 0x0FFF, "transport idx out of range: {}", t);
  }

  explicit mode_id(nigiri::transport_mode_id_t const x) {
    std::memcpy(this, &x, sizeof(mode_id));
  }

  nigiri::stop_idx_t get_stop() const {
    return static_cast<nigiri::stop_idx_t>(stop_idx_);
  }

  nigiri::flex_transport_idx_t get_flex_transport() const {
    return nigiri::flex_transport_idx_t{transport_};
  }

  nigiri::transport_mode_id_t to_id() {
    static_assert(sizeof(mode_id) == sizeof(nigiri::transport_mode_id_t));
    auto id = nigiri::transport_mode_id_t{};
    std::memcpy(&id, this, sizeof(id));
    return id;
  }

  nigiri::flex_transport_idx_t::value_t stop_idx_ : 8;
  nigiri::flex_transport_idx_t::value_t transport_ : 24;
};

}  // namespace motis::flex