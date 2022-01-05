#pragma once

#include <map>
#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/schedule.h"
#include "motis/schedule-format/Schedule_generated.h"

namespace motis::loader {

struct graph_builder;

mcd::hash_map<Station const*, station_node*> build_stations(
    graph_builder&, std::vector<Schedule const*> const&, bool use_platforms,
    bool no_local_stations);

}  // namespace motis::loader
