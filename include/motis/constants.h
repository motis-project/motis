#pragma once

#include <cstddef>

namespace motis {

// Current /plan API version for query generation.
constexpr auto const kPlanPath = "/api/v6/plan";

// search radius for neighbors to route to [meters]
constexpr auto const kMaxDistance = 2000;

// upper bound for radius API parameters [meters]
constexpr auto const kMaxQueryRadius = 10'000;

// upper bound for the /plan radius API parameter [meters]
constexpr auto const kMaxPlanRadius = 500'000.0;

// upper bound for the numLegAlternatives API parameter
constexpr auto const kMaxLegAlternatives = std::size_t{5U};

// max distance from start/destination coordinate to way segment [meters]
constexpr auto const kMaxMatchingDistance = 25.0;

constexpr auto const kMaxWheelchairMatchingDistance = 8.0;

// max distance from gbfs vehicle/station to way segment [meters]
constexpr auto const kMaxGbfsMatchingDistance = 100.0;

// multiplier for transfer times
constexpr auto const kTransferTimeMultiplier = 1.5F;

// footpaths of public transport locations around this distance
// are updated on elevator status changes [meters]
constexpr auto const kElevatorUpdateRadius = 1000.;

}  // namespace motis
