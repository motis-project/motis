#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nigiri/types.h"

#include "motis/rt/vehicle_prediction.h"

namespace transit_realtime {
class FeedMessage;
class TripDescriptor;
}  // namespace transit_realtime

namespace motis {

struct scheduled_provider_stop {
  unsigned static_stop_sequence_{};
  std::optional<std::int64_t> arrival_timestamp_seconds_;
  std::optional<std::int64_t> departure_timestamp_seconds_;
};

struct resolved_provider_trip {
  nigiri::transport transport_{nigiri::transport::invalid()};
  std::vector<scheduled_provider_stop> stops_;
};

struct provider_stop_timing {
  unsigned static_stop_sequence_{};
  std::optional<std::int64_t> arrival_timestamp_seconds_;
  std::optional<std::int64_t> departure_timestamp_seconds_;
  std::optional<std::int64_t> arrival_delay_seconds_;
  std::optional<std::int64_t> departure_delay_seconds_;
};

struct provider_timing_candidate {
  nigiri::transport transport_{nigiri::transport::invalid()};
  std::optional<std::int64_t> feed_timestamp_seconds_;
  std::optional<std::int64_t> trip_update_timestamp_seconds_;
  std::vector<provider_stop_timing> stops_;
};

struct provider_timing_extraction {
  std::vector<provider_timing_candidate> candidates_;
  std::size_t operational_only_trip_updates_{};
  std::size_t unresolved_trip_updates_{};
};

using provider_trip_resolver =
    std::function<std::optional<resolved_provider_trip>(
        transit_realtime::TripDescriptor const&)>;

// Reads timing fields only. The source message stays the authority for all
// operational semantics and is never mutated by extraction or selection.
[[nodiscard]] provider_timing_extraction extract_provider_timing(
    transit_realtime::FeedMessage const&, provider_trip_resolver const&);

enum class vehicle_prediction_source { kProvider, kGps, kSchedule };

enum class vehicle_prediction_selection_reason {
  kProviderOnly,
  kGpsOnly,
  kProviderHigherConfidence,
  kGpsHigherConfidence,
  kProviderProgressInconsistent,
  kProviderRecoveryPending,
  kProviderRecovered,
  kSourceHysteresis,
  kNoUsableCandidate,
  kPolicyUnavailable
};

enum class timing_candidate_rejection_reason {
  kStale,
  kLowConfidence,
  kPhysicallyUnreachable,
  kTimestampNotComparable,
  kProgressNotComparable,
  kProgressInconsistent
};

[[nodiscard]] char const* to_str(vehicle_prediction_source);
[[nodiscard]] char const* to_str(vehicle_prediction_selection_reason);
[[nodiscard]] char const* to_str(timing_candidate_rejection_reason);

struct timing_source_candidate {
  vehicle_prediction_source source_{vehicle_prediction_source::kSchedule};
  std::int64_t reference_timestamp_seconds_{};
  double confidence_{};
  bool physically_reachable_{false};
  std::optional<double> implied_progress_m_;
  std::vector<vehicle_stop_prediction> predictions_;
};

struct vehicle_prediction_selector_policy {
  std::int64_t max_candidate_age_seconds_{};
  std::int64_t max_timestamp_skew_seconds_{};
  double max_progress_difference_m_{};
  double min_provider_confidence_{};
  double min_gps_confidence_{};
  double min_source_switch_confidence_advantage_{};
  std::int64_t state_ttl_seconds_{};
  std::int64_t flap_window_seconds_{};
  std::int64_t minute_boundary_hysteresis_seconds_{};

  [[nodiscard]] bool valid() const;
};

struct vehicle_prediction_selection_input {
  nigiri::transport transport_{nigiri::transport::invalid()};
  std::int64_t now_seconds_{};
  std::optional<timing_source_candidate> provider_;
  std::optional<timing_source_candidate> gps_;
};

struct vehicle_prediction_selection_diagnostics {
  struct rendered_stop_delay {
    unsigned static_stop_sequence_{};
    std::int64_t raw_delay_seconds_{};
    std::int64_t rendered_delay_minutes_{};
  };

  std::optional<timing_candidate_rejection_reason> provider_rejection_;
  std::optional<timing_candidate_rejection_reason> gps_rejection_;
  std::optional<std::int64_t> candidate_timestamp_skew_seconds_;
  std::optional<double> progress_difference_m_;
  std::optional<double> selected_confidence_;
  bool source_transition_{false};
  bool provider_recovery_{false};
  bool flap_{false};
  unsigned provider_consistent_cycles_{};
  std::vector<rendered_stop_delay> rendered_delays_;
};

struct vehicle_prediction_selection {
  vehicle_prediction_source source_{vehicle_prediction_source::kSchedule};
  vehicle_prediction_selection_reason reason_{
      vehicle_prediction_selection_reason::kNoUsableCandidate};
  std::vector<vehicle_stop_prediction> predictions_;
  vehicle_prediction_selection_diagnostics diagnostics_;
};

[[nodiscard]] vehicle_prediction_selection select_vehicle_prediction_source(
    vehicle_prediction_selection_input const&,
    vehicle_prediction_selector_policy const&);

struct vehicle_prediction_selection_state {
  struct entry {
    nigiri::transport transport_{nigiri::transport::invalid()};
    vehicle_prediction_source selected_{vehicle_prediction_source::kSchedule};
    std::int64_t last_seen_seconds_{};
    std::optional<std::int64_t> last_transition_seconds_;
    unsigned provider_consistent_cycles_{};
    bool provider_rejected_{false};
    std::vector<std::pair<unsigned, std::int64_t>> rendered_minutes_;
  };

  [[nodiscard]] vehicle_prediction_selection select(
      vehicle_prediction_selection_input const&,
      vehicle_prediction_selector_policy const&,
      bool completed = false);
  void expire(std::int64_t now_seconds,
              vehicle_prediction_selector_policy const&);
  [[nodiscard]] std::size_t size() const { return entries_.size(); }

private:
  std::vector<entry> entries_;
};

// Uses C++ integer division (truncate toward zero). Hysteresis affects only the
// rendered minute; raw seconds remain in selection diagnostics.
[[nodiscard]] std::int64_t rendered_delay_minutes(std::int64_t raw_seconds);
[[nodiscard]] std::int64_t rendered_delay_minutes_with_hysteresis(
    std::int64_t raw_seconds,
    std::optional<std::int64_t> previous_rendered_minutes,
    std::int64_t boundary_hysteresis_seconds);

struct config;

struct vehicle_prediction_shadow_request {
  std::string feed_;
  nigiri::clasz mode_{nigiri::clasz::kOther};
  vehicle_prediction_selection_input selection_;
  bool completed_{false};
};

using vehicle_prediction_selector_policy_resolver =
    std::function<std::optional<vehicle_prediction_selector_policy>(
        std::string_view, nigiri::clasz)>;

struct vehicle_prediction_shadow_summary {
  std::string feed_;
  nigiri::clasz mode_{nigiri::clasz::kOther};
  std::size_t evaluated_{};
  std::size_t provider_selected_{};
  std::size_t gps_selected_{};
  std::size_t schedule_fallback_{};
  std::size_t transitions_{};
  std::size_t provider_recoveries_{};
  std::size_t rejections_{};
  std::size_t flaps_{};

  [[nodiscard]] double flap_rate() const;
};

struct vehicle_prediction_shadow_cycle_result {
  std::vector<vehicle_prediction_selection> selections_;
  std::vector<vehicle_prediction_shadow_summary> summaries_;
};

// Shadow evaluation owns no provider message and cannot affect API responses.
// Feed/mode enablement is resolved for every request independently.
[[nodiscard]] vehicle_prediction_shadow_cycle_result
evaluate_vehicle_prediction_shadow_cycle(
    config const&,
    std::span<vehicle_prediction_shadow_request const>,
    vehicle_prediction_selector_policy_resolver const&,
    vehicle_prediction_selection_state&);

// Emits one aggregate line per feed/mode for the completed cycle.
void log_vehicle_prediction_shadow_cycle(
    std::span<vehicle_prediction_shadow_summary const>);

}  // namespace motis
