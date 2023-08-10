#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "motis/core/schedule/schedule.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/trip_formation.h"

namespace motis::paxmon {

trip_formation const* get_trip_formation(capacity_maps const& caps,
                                         trip const* trp);

std::pair<trip_formation const*, trip_formation_section const*>
get_trip_formation_section(schedule const& sched, capacity_maps const& caps,
                           trip const* trp, ev_key const& ev_key_from);

inline capacity_source get_worst_source(capacity_source const a,
                                        capacity_source const b) {
  return static_cast<capacity_source>(
      std::max(static_cast<std::underlying_type_t<capacity_source>>(a),
               static_cast<std::underlying_type_t<capacity_source>>(b)));
}

inline std::uint16_t clamp_capacity(capacity_maps const& caps,
                                    std::uint16_t const capacity) {
  return std::max(caps.min_capacity_, capacity);
}

section_capacity get_section_capacity(schedule const& sched,
                                      capacity_maps const& caps,
                                      light_connection const& lc,
                                      ev_key const& ev_key_from,
                                      bool const detailed);

std::optional<detailed_capacity> get_override_capacity(
    schedule const& sched, capacity_maps const& caps, trip const* trp,
    ev_key const& ev_key_from);

}  // namespace motis::paxmon
