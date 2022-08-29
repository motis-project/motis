#pragma once

#include <chrono>

#include "nigiri/types.h"

#include "motis/core/common/unixtime.h"

namespace motis::nigiri {

inline unixtime to_motis_unixtime(::nigiri::unixtime_t const t) {
  return motis::unixtime{
      std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch())
          .count()};
}

}  // namespace motis::nigiri