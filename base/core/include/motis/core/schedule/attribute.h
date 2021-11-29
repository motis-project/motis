#pragma once

#include "motis/core/schedule/bitfield.h"
#include "motis/string.h"

namespace motis {

struct attribute {
  mcd::string text_;
  mcd::string code_;
};

struct traffic_day_attribute {
  bitfield_idx_or_ptr traffic_days_;
  ptr<attribute> const attr_;
};

}  // namespace motis
