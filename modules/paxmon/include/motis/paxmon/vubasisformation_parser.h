#pragma once

#include <string_view>

#include "motis/paxmon/capacity.h"

namespace motis::paxmon {

void parse_vubasisformation(std::string_view msg, capacity_maps& caps);

}  // namespace motis::paxmon
