#pragma once

#include "cista/reflection/comparable.h"

#include "motis/string.h"

namespace motis {

struct free_text {
  CISTA_COMPARABLE()
  int32_t code_{};
  mcd::string text_;
  mcd::string type_;
};

}  // namespace motis
