#pragma once

#include "cista/reflection/comparable.h"

#include "motis/core/schedule/bitfield.h"
#include "motis/string.h"

namespace motis {

struct attribute {
  CISTA_COMPARABLE()
  mcd::string code_;
  mcd::string text_;
};

struct traffic_day_attribute {
  bitfield_idx_or_ptr traffic_days_;
  ptr<attribute> const attr_;
};

}  // namespace motis
