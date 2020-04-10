#pragma once

#include "motis/path/prepare/schedule/stations.h"

namespace motis::path {

void find_stop_positions(std::string const& osm_file,
                         std::string const& sched_path,
                         std::vector<station>& stations);

}  // namespace motis::path
