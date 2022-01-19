#pragma once

#include <ctime>

#include "motis/string.h"

#include "motis/core/common/hash_helper.h"
#include "motis/core/common/unixtime.h"

namespace motis {

struct extern_trip {
  CISTA_COMPARABLE()

  mcd::string station_id_;
  uint32_t train_nr_{0};
  unixtime time_{0};

  mcd::string target_station_id_;
  unixtime target_time_{0};
  mcd::string line_id_;
};

}  // namespace motis
