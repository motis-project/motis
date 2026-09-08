#pragma once

#include "osr/types.h"

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace motis::flex {

struct mode_id {
  mode_id(nigiri::flex_transport_idx_t const t,
          nigiri::stop_idx_t const stop_idx,
          osr::direction const dir)
      : transport_{t},
        dir_{dir != osr::direction::kForward},
        stop_idx_{stop_idx} {}

  explicit mode_id(nigiri::routing::transport_mode_t::payload_t const x) {
    std::memcpy(this, &x, sizeof(mode_id));
  }

  osr::direction get_dir() const {
    return dir_ == 0 ? osr::direction::kForward : osr::direction::kBackward;
  }

  nigiri::stop_idx_t get_stop() const {
    return static_cast<nigiri::stop_idx_t>(stop_idx_);
  }

  nigiri::flex_transport_idx_t get_flex_transport() const {
    return nigiri::flex_transport_idx_t{transport_};
  }

  nigiri::routing::transport_mode_t::payload_t to_id() const {
    static_assert(sizeof(mode_id) ==
                  sizeof(nigiri::routing::transport_mode_t::payload_t));
    auto id = nigiri::routing::transport_mode_t::payload_t{};
    std::memcpy(&id, this, sizeof(id));
    return id;
  }

  nigiri::flex_transport_idx_t::value_t transport_ : 24;
  nigiri::flex_transport_idx_t::value_t dir_ : 1;
  nigiri::flex_transport_idx_t::value_t stop_idx_ : 7;
};

}  // namespace motis::flex