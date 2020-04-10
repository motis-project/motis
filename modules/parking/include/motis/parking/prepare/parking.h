#pragma once

#include <string>
#include <vector>

#include "motis/parking/parking_lot.h"

namespace motis::parking::prepare {

bool extract_parkings(std::string const& osm_file,
                      std::string const& parking_file,
                      std::vector<motis::parking::parking_lot>& parkings);

}  // namespace motis::parking::prepare
