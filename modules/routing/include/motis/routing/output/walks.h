#pragma once

#include <vector>

#include "motis/routing/output/stop.h"
#include "motis/routing/output/transport.h"

namespace motis::routing::output {

void update_walk_times(std::vector<intermediate::stop>& stops,
                       std::vector<intermediate::transport> const& transports);

}  // namespace motis::routing::output
