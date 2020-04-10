#pragma once

#include "motis/path/prepare/osm/osm_way.h"

namespace motis::path {

std::vector<std::vector<osm_way>> load_osm_ways(std::string const& filename);

void store_osm_ways(std::string const& filename,
                    std::vector<std::vector<osm_way>> const&);

}  // namespace motis::path
