#pragma once

#include <vector>

#include "motis/hash_map.h"

#include "motis/core/schedule/schedule.h"

namespace motis::loader {

struct Station;  // NOLINT
struct Schedule;  // NOLINT

mcd::hash_map<Station const*, station_node*> build_stations(
    schedule&, std::vector<Schedule const*> const&, bool no_local_stations);

}  // namespace motis::loader
