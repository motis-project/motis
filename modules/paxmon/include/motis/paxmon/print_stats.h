#pragma once

#include "motis/paxmon/graph.h"
#include "motis/paxmon/statistics.h"

namespace motis::paxmon {

void print_graph_stats(graph_statistics const& graph_stats);
void print_allocator_stats(graph const& g);

}  // namespace motis::paxmon
