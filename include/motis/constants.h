#pragma once

#include "cista/serialization.h"

namespace motis {

// search radius for neighbors to route to [seconds]
constexpr auto const kMaxDuration = 15 * 60;

// search radius for neighbors to route to [meters]
constexpr auto const kMaxDistance = 2000;

// max distance from start/destination coordinate to way segment [meters]
constexpr auto const kMaxMatchingDistance = 25;

// distance between location in timetable and OSM platform coordinate [meters]
constexpr auto const kMaxAdjust = 200;

// multiplier for transfer times
constexpr auto const kTransferTimeMultiplier = 1.5F;

// footpaths of public transport locations around this distance
// are updated on elevator status changes [meters]
constexpr auto const kElevatorUpdateRadius = 1000.;

// first nigiri::transport_mode_id_t used for GBFS
constexpr auto const kGbfsTransportModeIdOffset = 10;

}  // namespace motis
