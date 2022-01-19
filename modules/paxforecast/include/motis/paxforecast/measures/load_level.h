#pragma once

#include <cstdint>

namespace motis::paxforecast::measures {

enum load_level : std::uint8_t { UNKNOWN, LOW, NO_SEATS, FULL };

}  // namespace motis::paxforecast::measures
