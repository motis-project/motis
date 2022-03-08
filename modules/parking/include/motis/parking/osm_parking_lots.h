#pragma once

#include <string>
#include <vector>

#include "motis/parking/parking_lot.h"

namespace motis::parking {

std::vector<parking_lot> extract_osm_parking_lots(std::string const& osm_file);

}  // namespace motis::parking
