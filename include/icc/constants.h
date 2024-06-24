#pragma once

namespace icc {

// search radius for neighbors to route to [seconds]
constexpr auto const kMaxDuration = 15 * 60;

// search radius for neighbors to route to [meters]
constexpr auto const kMaxDistance = 2000;

// maximal distance from routing start/destination coordinate to way segment
constexpr auto const kMaxMatchingDistance = 8;

// meters distance between location in timetable and OSM platform coordinate
constexpr auto const kMaxAdjust = 45;

// multiplier for transfer times
constexpr auto const kTransferTimeMultiplier = 2;

}  // namespace icc