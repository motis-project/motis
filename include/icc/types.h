#pragma once

#include <cinttypes>

#include "geo/latlng.h"

#include "cista/strong.h"

namespace icc {

enum class status : bool { kActive, kInactive };

using elevator_idx_t = cista::strong<std::uint32_t, struct elevator_idx_>;

struct elevator {
  std::int64_t id_;
  geo::latlng pos_;
  status status_;
  std::string desc_;
};

}  // namespace icc