#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace motis {

struct config;
struct tag_lookup;
struct vehicle_observation_history;

namespace vehicle_positions {
struct vehicle_position_store;
}

enum class trip_progress_diagnostic_status {
  kProjected,
  kStale,
  kMissingTripId,
  kUnsupportedScheduleRelationship,
  kUnresolvedTrip,
  kUnscheduledTrip,
  kMissingHistory,
  kMissingShape,
  kOffShape,
  kAmbiguous,
  kImplausible
};

struct trip_progress_diagnostic {
  std::string feed_;
  std::optional<nigiri::clasz> mode_;
  trip_progress_diagnostic_status status_{
      trip_progress_diagnostic_status::kUnresolvedTrip};
  std::optional<double> lateral_error_m_;
};

[[nodiscard]] char const* to_str(trip_progress_diagnostic_status);

// Evaluates current vehicle observations without mutating the realtime
// timetable or any published state. Vehicles configured as `off` are omitted.
[[nodiscard]] std::vector<trip_progress_diagnostic>
evaluate_trip_progress_diagnostics(
    config const&,
    tag_lookup const&,
    nigiri::timetable const&,
    nigiri::rt_timetable const*,
    nigiri::shapes_storage const*,
    vehicle_positions::vehicle_position_store const&,
    vehicle_observation_history const&,
    std::int64_t now);

}  // namespace motis
