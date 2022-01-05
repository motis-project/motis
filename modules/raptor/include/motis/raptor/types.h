#pragma once

#include <cstdint>
#include <vector>
#include <limits>

#include "motis/core/schedule/time.h"

namespace motis::raptor {
using time8 = uint8_t;

using stop_id = int32_t;
using route_id = uint32_t;
using footpath_id = int32_t;
using trip_id = uint32_t;
using arrival_id = uint32_t;

using trait_id = uint8_t;
using dimension_id = uint8_t;

using motis_id = int32_t;

// these are only 16bit wide, because they are used relativ to a station/route
// i.e. how many stops/trips a single route has
//      how many routes/footpaths a single station has
using trip_count = uint16_t;
using stop_count = uint16_t;
using route_count = uint16_t;
using footpath_count = uint16_t;

using stop_offset = uint16_t;

using stop_times_index = uint32_t;
using route_stops_index = uint32_t;
using stop_routes_index = uint32_t;
using footpaths_index = uint32_t;

using raptor_round = uint8_t;
using transfers = uint8_t;

//additional attributes
using occ_t = uint8_t;
using transfer_class_t = uint8_t;

using earliest_arrivals = std::vector<time>;

template <typename T>
inline constexpr T min_value = std::numeric_limits<T>::min();

template <typename T>
inline constexpr T max_value = std::numeric_limits<T>::max();

template <typename T>
inline constexpr T invalid = max_value<T>;
// Template specializations in raptor_timetable.cc

template <typename T>
inline constexpr auto valid(T const& value) {
  return value != invalid<T>;
}
}