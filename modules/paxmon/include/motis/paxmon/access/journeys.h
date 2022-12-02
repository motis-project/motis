#pragma once

#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

fws_compact_journey add_compact_journey(universe& uv,
                                        compact_journey const& cj);

}  // namespace motis::paxmon
