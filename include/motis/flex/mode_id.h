#pragma once

#include "osr/types.h"

#include "nigiri/types.h"

namespace motis::flex {

struct mode_id {
  mode_id(nigiri::flex_transport_idx_t const t,
          nigiri::stop_idx_t const stop_idx,
          osr::direction const dir)
      : transport_{t},
        dir_{dir != osr::direction::kForward},
        stop_idx_{stop_idx},
        msb_{1U} {}

  static bool is_flex(nigiri::transport_mode_id_t const x) {
    return (x & 0x80'00'00'00) == 0x80'00'00'00;
  }

  explicit mode_id(nigiri::transport_mode_id_t const x) {
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

  nigiri::transport_mode_id_t to_id() const {
    static_assert(sizeof(mode_id) == sizeof(nigiri::transport_mode_id_t));
    auto id = nigiri::transport_mode_id_t{};
    std::memcpy(&id, this, sizeof(id));
    return id;
  }

  nigiri::flex_transport_idx_t::value_t transport_ : 23;
  nigiri::flex_transport_idx_t::value_t dir_ : 1;
  nigiri::flex_transport_idx_t::value_t stop_idx_ : 7;
  nigiri::flex_transport_idx_t::value_t msb_ : 1;
};

}  // namespace motis::flex