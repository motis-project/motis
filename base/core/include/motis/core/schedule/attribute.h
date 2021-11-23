#pragma once

#include "cista/reflection/comparable.h"

#include "motis/string.h"

namespace motis {

struct attribute {
  CISTA_COMPARABLE()
  mcd::string code_;
  mcd::string text_;
};

}  // namespace motis
