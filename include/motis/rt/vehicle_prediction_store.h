#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nigiri/types.h"

#include "motis/rt/vehicle_prediction_selection.h"

namespace motis {

struct prediction_candidate_diagnostic {
  vehicle_prediction_source source_{vehicle_prediction_source::kSchedule};
  std::int64_t predicted_timestamp_seconds_{};
  std::int64_t delay_seconds_{};
  std::optional<double> confidence_;
  std::optional<std::int64_t> reference_timestamp_seconds_;
};

struct vehicle_prediction_diagnostic_entry {
  nigiri::transport transport_{nigiri::transport::invalid()};
  unsigned static_stop_sequence_{};
  std::string trip_id_;
  std::int64_t observed_at_seconds_{};
  std::optional<prediction_candidate_diagnostic> provider_;
  std::optional<prediction_candidate_diagnostic> gps_;
  prediction_candidate_diagnostic effective_;
  vehicle_prediction_source selected_source_{
      vehicle_prediction_source::kSchedule};
  vehicle_prediction_selection_reason selection_reason_{
      vehicle_prediction_selection_reason::kNoUsableCandidate};
  std::optional<timing_candidate_rejection_reason> provider_rejection_;
  std::optional<timing_candidate_rejection_reason> gps_rejection_;
  std::optional<double> selected_confidence_;
  std::optional<std::int64_t> candidate_timestamp_skew_seconds_;
  std::optional<double> projection_error_m_;
  std::optional<double> progress_difference_m_;
  bool source_transition_{false};
  bool provider_recovery_{false};
  bool flap_{false};
  unsigned provider_consistent_cycles_{};
};

// Immutable and owned by the same published RT snapshot as the timetable it
// describes. Entries are sorted by scheduled trip instance and GTFS stop
// sequence, which makes endpoint lookups deterministic and allocation-free.
struct vehicle_prediction_diagnostics_store {
  struct limits {
    std::int64_t max_age_seconds_{300};
    std::size_t max_entries_{100'000U};
  };

  [[nodiscard]] static std::unique_ptr<vehicle_prediction_diagnostics_store>
  build(bool enabled,
        std::vector<vehicle_prediction_diagnostic_entry>,
        std::int64_t now_seconds);
  [[nodiscard]] static std::unique_ptr<vehicle_prediction_diagnostics_store>
  build(bool enabled,
        std::vector<vehicle_prediction_diagnostic_entry>,
        std::int64_t now_seconds,
        limits);

  [[nodiscard]] vehicle_prediction_diagnostic_entry const* find(
      nigiri::transport, unsigned static_stop_sequence) const;
  [[nodiscard]] vehicle_prediction_diagnostic_entry const* find_event(
      nigiri::transport,
      std::string_view trip_id,
      std::int64_t scheduled_timestamp_seconds) const;
  [[nodiscard]] std::size_t size() const { return entries_.size(); }

  std::vector<vehicle_prediction_diagnostic_entry> entries_;
};

}  // namespace motis
