#pragma once

#include <cinttypes>

#include "motis/string.h"

namespace motis {

struct category {
  mcd::string name_;
  uint8_t output_rule_;
};

}  // namespace motis
