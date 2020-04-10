#pragma once

#include <string>
#include <vector>

#include "motis/path/prepare/osm/osm_way.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

std::vector<std::vector<osm_way>> parse_network(
    std::string const& osm_file, source_spec::category const& category);

}  // namespace motis::path
