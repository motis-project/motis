#pragma once

#include <cstdint>
#include <utility>

namespace motis::paxmon {

enum class capacity_source : std::uint16_t {
  TRIP_EXACT,
  TRAIN_NR,
  CATEGORY,
  CLASZ,
  DEFAULT
};

inline std::uint16_t encode_capacity(std::uint16_t capacity,
                                     capacity_source source) {
  return (static_cast<std::uint16_t>(source) << 13) | capacity;
}

inline std::uint16_t encode_capacity(
    std::pair<std::uint16_t, capacity_source> const& p) {
  return encode_capacity(p.first, p.second);
}

inline std::uint16_t get_capacity(std::uint16_t encoded) {
  return encoded & 0x1FFF;
}

inline capacity_source get_capacity_source(std::uint16_t encoded) {
  return static_cast<capacity_source>((encoded >> 13) & 0x7);
}

inline std::pair<std::uint16_t, capacity_source> decode_capacity(
    std::uint16_t encoded) {
  return {get_capacity(encoded), get_capacity_source(encoded)};
}

}  // namespace motis::paxmon
