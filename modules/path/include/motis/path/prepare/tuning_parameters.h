#pragma once

namespace motis::path {

// Matching Distance from station (schedule) to stop sosition (OSM)
constexpr auto const kStationStopDistance = 100;  // [m]
constexpr auto const kStationStopFallbackDistance = 250;  // [m]

// Matching distance to OSM data (from station / stop position)
constexpr auto const kMaxMatchDistance = 1000;  // [m]
constexpr auto const kStationMatchDistance = 50;  // [m]
constexpr auto const kStopMatchDistance = 20;  // [m]

//  Weights for data from different sources
constexpr auto const kOsmRelationBonusFactor = 0.5;
constexpr auto const kStubStrategyPenaltyFactor = 100;
constexpr auto const kMinorMismatchPenaltyFactor = 1.5;
constexpr auto const kMajorMismatchPenaltyFactor = 2;
constexpr auto const kUnlikelyPenaltyFactor = 5;
constexpr auto const kVeryUnlikelyPenaltyFactor = 10;

}  // namespace motis::path
