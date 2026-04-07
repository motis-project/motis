#pragma once

#include <filesystem>

#include "motis/config.h"
#include "motis/data.h"

namespace motis {

void import_route_tiles(config const&, data&, std::filesystem::path const&);

}  // namespace motis
