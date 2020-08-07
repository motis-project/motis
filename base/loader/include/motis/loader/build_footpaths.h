#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/schedule.h"

#include "motis/loader/loader_options.h"

namespace motis::loader {

struct Schedule;  // NOLINT
struct Station;  // NOLINT

void build_footpaths(schedule&, loader_options const&,
                     mcd::hash_map<Station const*, station_node*> const&,
                     std::vector<Schedule const*> const&);

}  // namespace motis::loader
