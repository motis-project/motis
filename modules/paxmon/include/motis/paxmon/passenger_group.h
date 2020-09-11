#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/paxmon/compact_journey.h"

namespace motis::paxmon {

struct edge;

struct data_source {
  std::uint64_t primary_ref_{};
  std::uint64_t secondary_ref_{};
};

enum class group_source_flags : std::uint8_t {
  NONE = 0,
  MATCH_INEXACT_TIME = 1U << 0U,
  MATCH_JOURNEY_SUBSET = 1U << 1U,
  MATCH_REROUTED = 1U << 2U,
  FORECAST = 1U << 3U
};

inline constexpr group_source_flags operator|(group_source_flags const& a,
                                              group_source_flags const& b) {
  return static_cast<group_source_flags>(
      static_cast<std::underlying_type_t<group_source_flags>>(a) |
      static_cast<std::underlying_type_t<group_source_flags>>(b));
}

inline constexpr group_source_flags operator&(group_source_flags const& a,
                                              group_source_flags const& b) {
  return static_cast<group_source_flags>(
      static_cast<std::underlying_type_t<group_source_flags>>(a) &
      static_cast<std::underlying_type_t<group_source_flags>>(b));
}

inline constexpr group_source_flags operator|=(group_source_flags& a,
                                               group_source_flags const& b) {
  a = a | b;
  return a;
}

inline constexpr group_source_flags operator&=(group_source_flags& a,
                                               group_source_flags const& b) {
  a = a & b;
  return a;
}

struct passenger_group {
  compact_journey compact_planned_journey_;
  std::uint64_t id_{};
  data_source source_{};
  std::uint16_t passengers_{1};
  motis::time planned_arrival_time_{INVALID_TIME};
  group_source_flags source_flags_{group_source_flags::NONE};
  bool ok_{true};
  float probability_{1.0};
  std::vector<edge*> edges_{};
};

}  // namespace motis::paxmon
