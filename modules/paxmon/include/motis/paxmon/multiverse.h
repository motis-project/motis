#pragma once

#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct multiverse {
  universe& primary() { return primary_; }

  universe primary_;
};

}  // namespace motis::paxmon
