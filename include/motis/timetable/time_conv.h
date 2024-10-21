#pragma once

#include <chrono>
#include <cinttypes>

#include "nigiri/types.h"

namespace motis {

inline std::int64_t to_seconds(nigiri::unixtime_t const t) {
  return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch())
      .count();
}

inline std::int64_t to_seconds(nigiri::i32_minutes const t) {
  return std::chrono::duration_cast<std::chrono::seconds>(t).count();
}

inline std::int64_t to_ms(nigiri::i32_minutes const t) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(t).count();
}

}  // namespace motis