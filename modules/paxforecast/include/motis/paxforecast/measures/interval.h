#pragma once

#include "motis/core/schedule/time.h"

namespace motis::paxforecast::measures {

struct interval {
  time begin_{INVALID_TIME};
  time end_{INVALID_TIME};
};

}  // namespace motis::paxforecast::measures
