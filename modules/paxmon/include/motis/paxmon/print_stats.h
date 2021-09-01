#pragma once

#include "motis/paxmon/statistics.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

void print_graph_stats(graph_statistics const& graph_stats);
void print_allocator_stats(universe const& uv);

}  // namespace motis::paxmon
