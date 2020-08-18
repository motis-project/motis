#pragma once

#include <cstdint>
#include <utility>

namespace motis::paxmon {

enum class capacity_source : std::uint16_t {
  TRIP_EXACT,
  TRAIN_NR,
  CATEGORY,
  CLASZ,
  SPECIAL
};

inline constexpr std::uint16_t encode_capacity(std::uint16_t capacity,
                                               capacity_source source) {
  return (static_cast<std::uint16_t>(source) << 13) | capacity;
}

inline constexpr std::uint16_t encode_capacity(
    std::pair<std::uint16_t, capacity_source> const& p) {
  return encode_capacity(p.first, p.second);
}

inline constexpr std::uint16_t get_capacity(std::uint16_t encoded) {
  return encoded & 0x1FFF;
}

inline constexpr capacity_source get_capacity_source(std::uint16_t encoded) {
  return static_cast<capacity_source>((encoded >> 13) & 0x7);
}

inline constexpr std::pair<std::uint16_t, capacity_source> decode_capacity(
    std::uint16_t encoded) {
  return {get_capacity(encoded), get_capacity_source(encoded)};
}

constexpr const std::uint16_t UNKNOWN_CAPACITY = 0;
constexpr const std::uint16_t UNKNOWN_ENCODED_CAPACITY =
    encode_capacity(UNKNOWN_CAPACITY, capacity_source::SPECIAL);

constexpr const std::uint16_t UNLIMITED_CAPACITY = 0x1FFF;
constexpr const std::uint16_t UNLIMITED_ENCODED_CAPACITY =
    encode_capacity(UNLIMITED_CAPACITY, capacity_source::SPECIAL);

}  // namespace motis::paxmon
