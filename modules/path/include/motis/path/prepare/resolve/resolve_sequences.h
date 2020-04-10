#pragma once

#include <vector>

#include "motis/path/prepare/resolve/path_routing.h"
#include "motis/path/prepare/resolve/resolved_station_seq.h"
#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

std::vector<resolved_station_seq> resolve_sequences(
    std::vector<station_seq> const&, path_routing&);

}  // namespace motis::path
