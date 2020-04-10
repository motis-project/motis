#pragma once

#include "utl/struct/comparable.h"

#include "motis/string.h"

namespace motis {

struct free_text {
  MAKE_COMPARABLE()
  int32_t code_{};
  mcd::string text_;
  mcd::string type_;
};

}  // namespace motis
