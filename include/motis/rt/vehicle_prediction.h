#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "nigiri/types.h"

#include "motis/rt/vehicle_observation_history.h"

namespace nigiri::rt {
struct frun;
}

namespace nigiri {
struct shapes_storage;
}

namespace motis {

enum class vehicle_prediction_rejection_reason {
  kMissingTripId,
  kUnresolvedTrip,
  kUnscheduledTrip,
  kUnsupportedTripRelationship,
  kInsufficientHistory,
  kStaleHistory,
  kMissingShape,
  kOffShape,
  kAmbiguousProgress,
  kNonMonotonicProgress,
  kInvalidObservationTime,
  kImpossibleSpeed,
  kImpossibleTravelTime,
  kTerminal
};

[[nodiscard]] char const* to_str(vehicle_prediction_rejection_reason);

struct vehicle_prediction_confidence {
  double score_{};
  std::size_t observation_count_{};
  double lateral_error_m_{};
  double progress_velocity_mps_{};
  bool reported_speed_conflict_{false};
};

struct vehicle_stop_prediction {
  unsigned static_stop_sequence_{};
  std::int64_t scheduled_timestamp_seconds_{};
  std::int64_t predicted_timestamp_seconds_{};
  std::int64_t delay_seconds_{};
  std::int64_t horizon_seconds_{};
};

struct observed_stop_passage {
  unsigned static_stop_sequence_{};
  std::int64_t observed_timestamp_seconds_{};
  std::int64_t uncertainty_seconds_{};
};

struct vehicle_prediction_diagnostics {
  std::optional<vehicle_prediction_rejection_reason> rejection_;
  std::size_t fresh_observation_count_{};
  std::size_t uncertain_passage_count_{};
};

struct vehicle_prediction_batch {
  // V1 candidates are always keyed by the exact scheduled trip instance.
  nigiri::transport transport_{nigiri::transport::invalid()};
  std::optional<unsigned> delay_anchor_static_stop_sequence_;
  std::optional<std::int64_t> delay_anchor_seconds_;
  std::vector<vehicle_stop_prediction> predictions_;
  std::vector<observed_stop_passage> observed_passages_;
  std::optional<vehicle_prediction_confidence> confidence_;
  vehicle_prediction_diagnostics diagnostics_;

  [[nodiscard]] bool eligible() const {
    return !diagnostics_.rejection_.has_value() && !predictions_.empty();
  }
  [[nodiscard]] std::size_t estimated_memory_bytes() const;
};

struct vehicle_prediction_policy {
  std::int64_t max_observation_age_seconds_{300};
  std::size_t min_observations_{2U};
  double min_progress_velocity_mps_{0.5};
  double max_progress_velocity_mps_{55.0};
  std::int64_t max_predicted_travel_seconds_{4 * 60 * 60};
  std::int64_t max_passage_uncertainty_seconds_{90};
};

// Pure, shadow-safe evaluator. It does not retain history, mutate the run, or
// rewrite GTFS-RT. Projection, smoothing, schedule interpolation and GTFS stop
// sequence conversion remain behind this interface.
struct vehicle_prediction_engine {
  explicit vehicle_prediction_engine(nigiri::shapes_storage const&,
                                     vehicle_prediction_policy = {});
  ~vehicle_prediction_engine();

  vehicle_prediction_engine(vehicle_prediction_engine&&) noexcept;
  vehicle_prediction_engine& operator=(vehicle_prediction_engine&&) noexcept;
  vehicle_prediction_engine(vehicle_prediction_engine const&) = delete;
  vehicle_prediction_engine& operator=(vehicle_prediction_engine const&) =
      delete;

  [[nodiscard]] vehicle_prediction_batch evaluate(
      nigiri::rt::frun const&,
      std::span<vehicle_observation const>,
      std::int64_t now_seconds);

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis
