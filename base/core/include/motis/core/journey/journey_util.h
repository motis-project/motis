#pragma once

#include <cstdint>

namespace motis {
struct journey;

uint16_t get_duration(journey const&);
uint16_t get_transfers(journey const&);
uint16_t get_accessibility(journey const&);

}  // namespace motis
