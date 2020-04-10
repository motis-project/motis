#pragma once

#include "motis/core/schedule/nodes.h"

namespace motis {

node* build_route_node(int route_index, int node_id, station_node* station_node,
                       int transfer_time, bool in_allowed, bool out_allowed);

}  // namespace motis
