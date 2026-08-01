#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "geo/latlng.h"

namespace nigiri {
struct shapes_storage;
namespace rt {
struct frun;
}
}  // namespace nigiri

namespace motis {

enum class trip_progress_projection_status {
  kProjected,
  kMissingShape,
  kOffShape,
  kAmbiguous,
  kImplausible
};

enum class trip_progress_monotonicity {
  kNoPrior,
  kForward,
  kStationary,
  kMinorRegression
};

enum class vehicle_position_stop_status {
  kIncomingAt,
  kStoppedAt,
  kInTransitTo
};

struct vehicle_position_progress_constraint {
  unsigned current_static_stop_sequence_{};
  vehicle_position_stop_status status_{
      vehicle_position_stop_status::kInTransitTo};
};

struct trip_progress {
  double distance_along_shape_m_{};
  double lateral_error_m_{};
  unsigned next_static_stop_sequence_{};
  double distance_to_next_stop_m_{};
  trip_progress_monotonicity monotonicity_{
      trip_progress_monotonicity::kNoPrior};
};

struct trip_progress_projection {
  trip_progress_projection_status status_{
      trip_progress_projection_status::kMissingShape};
  std::optional<trip_progress> progress_;
};

struct trip_progress_stop {
  unsigned static_stop_sequence_{};
  double distance_along_shape_m_{};
  std::int64_t scheduled_arrival_time_{};
  std::int64_t scheduled_departure_time_{};
};

struct trip_progress_projector {
  explicit trip_progress_projector(nigiri::shapes_storage const&);
  ~trip_progress_projector();

  trip_progress_projector(trip_progress_projector&&) noexcept;
  trip_progress_projector& operator=(trip_progress_projector&&) noexcept;

  trip_progress_projector(trip_progress_projector const&) = delete;
  trip_progress_projector& operator=(trip_progress_projector const&) = delete;

  trip_progress_projection project(
      nigiri::rt::frun const&,
      geo::latlng const&,
      std::optional<trip_progress> const& = std::nullopt,
      std::optional<vehicle_position_progress_constraint> const& =
          std::nullopt);

  // Uses the same cached shape and stop-sequence conversion as project().
  // This keeps ETA consumers independent of shapes_storage internals.
  std::optional<std::vector<trip_progress_stop>> stop_timeline(
      nigiri::rt::frun const&);

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis
