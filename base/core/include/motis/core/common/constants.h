#pragma once

namespace motis {

constexpr const auto WALK_SPEED = 1.5;  // m/s
constexpr const auto BIKE_SPEED = 15.0 * (1000.0 / 3600.0);  // m/s (15 km/h)
constexpr const auto CAR_SPEED = 100.0 * (1000.0 / 3600.0);  // m/s (100 km/h)

constexpr const auto MAX_WALK_TIME = 10 * 60;  // s
constexpr const auto MAX_BIKE_TIME = 30 * 60;  // s

constexpr const auto MAX_WALK_DIST = MAX_WALK_TIME * WALK_SPEED;  // m
constexpr const auto MAX_BIKE_DIST = MAX_BIKE_TIME * BIKE_SPEED;  // m

constexpr const auto LINEAR_DIST_APPROX = 1.5;

constexpr const auto MAX_TRAVEL_TIME_HOURS = 24;
constexpr const auto MAX_TRAVEL_TIME_MINUTES = MAX_TRAVEL_TIME_HOURS * 60;
constexpr const auto MAX_TRAVEL_TIME_SECONDS = MAX_TRAVEL_TIME_MINUTES * 60;

constexpr const auto STATION_START = "START";
constexpr const auto STATION_END = "END";
constexpr const auto STATION_VIA0 = "VIA0";
constexpr const auto STATION_VIA1 = "VIA1";
constexpr const auto STATION_VIA2 = "VIA2";
constexpr const auto STATION_VIA3 = "VIA3";
constexpr const auto STATION_VIA4 = "VIA4";
constexpr const auto STATION_VIA5 = "VIA5";
constexpr const auto STATION_VIA6 = "VIA6";
constexpr const auto STATION_VIA7 = "VIA7";
constexpr const auto STATION_VIA8 = "VIA8";
constexpr const auto STATION_VIA9 = "VIA9";

}  // namespace motis
