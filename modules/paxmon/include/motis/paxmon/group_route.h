#pragma once

#include <cstdint>

#include "motis/core/schedule/time.h"

#include "motis/paxmon/index_types.h"

namespace motis::paxmon {

enum class route_source_flags : std::uint8_t {
  NONE = 0,
  MATCH_INEXACT_TIME = 1U << 0U,
  MATCH_JOURNEY_SUBSET = 1U << 1U,
  MATCH_REROUTED = 1U << 2U,
  FORECAST = 1U << 3U
};

inline constexpr route_source_flags operator|(route_source_flags const& a,
                                              route_source_flags const& b) {
  return static_cast<route_source_flags>(
      static_cast<std::underlying_type_t<route_source_flags>>(a) |
      static_cast<std::underlying_type_t<route_source_flags>>(b));
}

inline constexpr route_source_flags operator&(route_source_flags const& a,
                                              route_source_flags const& b) {
  return static_cast<route_source_flags>(
      static_cast<std::underlying_type_t<route_source_flags>>(a) &
      static_cast<std::underlying_type_t<route_source_flags>>(b));
}

inline constexpr route_source_flags operator|=(route_source_flags& a,
                                               route_source_flags const& b) {
  a = a | b;
  return a;
}

inline constexpr route_source_flags operator&=(route_source_flags& a,
                                               route_source_flags const& b) {
  a = a & b;
  return a;
}

struct group_route {
  inline motis::time estimated_arrival_time() const {
    return planned_arrival_time_ != INVALID_TIME
               ? static_cast<motis::time>(planned_arrival_time_ +
                                          estimated_delay_)
               : INVALID_TIME;
  }

  group_route_edges_index edges_index_{};
  float probability_{};
  compact_journey_index compact_journey_index_{};
  local_group_route_index local_group_route_index_{};
  motis::time planned_arrival_time_{INVALID_TIME};
  std::int16_t estimated_delay_{};
  bool broken_{};
  bool disabled_{};  // if true, route is not in the graph
  bool planned_{};
  bool destination_unreachable_{};
  route_source_flags source_flags_{route_source_flags::NONE};
};

inline group_route make_group_route(
    compact_journey_index const cj_index,
    local_group_route_index const lgr_index,
    group_route_edges_index const gre_index, float const probability,
    bool const planned,
    route_source_flags const source_flags = route_source_flags::NONE,
    motis::time const planned_arrival_time = INVALID_TIME,
    std::int16_t const estimated_delay = 0, bool const broken = false,
    bool const disabled = false, bool const destination_unreachable = false) {
  return group_route{gre_index,
                     probability,
                     cj_index,
                     lgr_index,
                     planned_arrival_time,
                     estimated_delay,
                     broken,
                     disabled,
                     planned,
                     destination_unreachable,
                     source_flags};
}

}  // namespace motis::paxmon
