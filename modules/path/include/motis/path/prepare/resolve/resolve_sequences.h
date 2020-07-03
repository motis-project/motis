#pragma once

#include <vector>

#include "motis/path/prepare/resolve/path_routing.h"
#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

mcd::vector<station_seq> resolve_sequences(mcd::vector<station_seq> const&,
                                           path_routing&);

}  // namespace motis::path
