#pragma once

#include "motis/core/schedule/time.h"

namespace motis {

struct interval {
  motis::time begin_{INVALID_TIME};
  motis::time end_{INVALID_TIME};
};

}  // namespace motis
