#include "motis/rt/vehicle_prediction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/rt/frun.h"
#include "nigiri/shapes_storage.h"

#include "motis/rt/trip_progress_projection.h"

namespace n = nigiri;

namespace motis {
namespace {

struct timed_progress {
  std::int64_t time_{};
  trip_progress progress_;
  std::optional<double> reported_speed_mps_;
  bool stopped_at_{false};
};

std::optional<vehicle_position_progress_constraint> constraint_for(
    vehicle_observation const& observation) {
  if (!observation.current_stop_sequence_.has_value()) {
    return std::nullopt;
  }
  auto status = vehicle_position_stop_status::kInTransitTo;
  if (observation.current_status_ == "INCOMING_AT") {
    status = vehicle_position_stop_status::kIncomingAt;
  } else if (observation.current_status_ == "STOPPED_AT") {
    status = vehicle_position_stop_status::kStoppedAt;
  }
  return vehicle_position_progress_constraint{
      .current_static_stop_sequence_ = *observation.current_stop_sequence_,
      .status_ = status};
}

vehicle_prediction_rejection_reason rejection_for(
    trip_progress_projection_status const status) {
  switch (status) {
    case trip_progress_projection_status::kMissingShape:
      return vehicle_prediction_rejection_reason::kMissingShape;
    case trip_progress_projection_status::kOffShape:
      return vehicle_prediction_rejection_reason::kOffShape;
    case trip_progress_projection_status::kAmbiguous:
      return vehicle_prediction_rejection_reason::kAmbiguousProgress;
    case trip_progress_projection_status::kImplausible:
      return vehicle_prediction_rejection_reason::kNonMonotonicProgress;
    case trip_progress_projection_status::kProjected: break;
  }
  std::unreachable();
}

double median(std::vector<double> values) {
  std::ranges::sort(values);
  auto const middle = values.size() / 2U;
  return values.size() % 2U == 0U ? (values[middle - 1U] + values[middle]) / 2.0
                                  : values[middle];
}

std::optional<double> robust_velocity(
    std::span<timed_progress const> progress) {
  auto velocities = std::vector<double>{};
  velocities.reserve(progress.size() - 1U);
  for (auto i = std::size_t{1U}; i != progress.size(); ++i) {
    auto const elapsed = progress[i].time_ - progress[i - 1U].time_;
    auto const distance = progress[i].progress_.distance_along_shape_m_ -
                          progress[i - 1U].progress_.distance_along_shape_m_;
    if (elapsed > 0 && distance > 0.0) {
      velocities.push_back(distance / static_cast<double>(elapsed));
    }
  }
  return velocities.empty() ? std::nullopt
                            : std::optional<double>{median(velocities)};
}

std::optional<double> reported_speed(std::span<timed_progress const> progress) {
  auto speeds = std::vector<double>{};
  for (auto const& sample : progress) {
    if (sample.reported_speed_mps_.has_value() &&
        std::isfinite(*sample.reported_speed_mps_) &&
        *sample.reported_speed_mps_ >= 0.0) {
      speeds.push_back(*sample.reported_speed_mps_);
    }
  }
  return speeds.empty() ? std::nullopt
                        : std::optional<double>{median(std::move(speeds))};
}

std::size_t next_stop_index(std::span<trip_progress_stop const> stops,
                            trip_progress const& progress) {
  auto const by_sequence =
      std::ranges::find(stops, progress.next_static_stop_sequence_,
                        &trip_progress_stop::static_stop_sequence_);
  if (by_sequence != end(stops)) {
    return static_cast<std::size_t>(std::distance(begin(stops), by_sequence));
  }
  auto const by_distance =
      std::ranges::lower_bound(stops, progress.distance_along_shape_m_, {},
                               &trip_progress_stop::distance_along_shape_m_);
  return static_cast<std::size_t>(std::distance(begin(stops), by_distance));
}

std::vector<observed_stop_passage> derive_passages(
    std::span<timed_progress const> samples,
    std::span<trip_progress_stop const> stops,
    vehicle_prediction_policy const& policy,
    std::size_t& uncertain_count) {
  auto facts = std::vector<observed_stop_passage>{};
  facts.reserve(stops.size());
  for (auto const& stop : stops) {
    for (auto i = std::size_t{1U}; i != samples.size(); ++i) {
      auto const& before = samples[i - 1U];
      auto const& after = samples[i];
      auto const elapsed = after.time_ - before.time_;
      auto const distance = after.progress_.distance_along_shape_m_ -
                            before.progress_.distance_along_shape_m_;
      if (elapsed <= 0 || distance <= 0.0 ||
          before.progress_.distance_along_shape_m_ >=
              stop.distance_along_shape_m_ ||
          after.progress_.distance_along_shape_m_ <
              stop.distance_along_shape_m_) {
        continue;
      }
      if (elapsed > policy.max_passage_uncertainty_seconds_) {
        ++uncertain_count;
        break;
      }
      auto const fraction = (stop.distance_along_shape_m_ -
                             before.progress_.distance_along_shape_m_) /
                            distance;
      facts.push_back({.static_stop_sequence_ = stop.static_stop_sequence_,
                       .observed_timestamp_seconds_ =
                           before.time_ + static_cast<std::int64_t>(
                                              std::llround(fraction * elapsed)),
                       .uncertainty_seconds_ = elapsed});
      break;
    }
  }
  return facts;
}

}  // namespace

struct vehicle_prediction_engine::impl {
  impl(n::shapes_storage const& shapes, vehicle_prediction_policy policy)
      : projector_{shapes}, policy_{std::move(policy)} {}

  trip_progress_projector projector_;
  vehicle_prediction_policy policy_;
};

char const* to_str(vehicle_prediction_rejection_reason const reason) {
  switch (reason) {
    case vehicle_prediction_rejection_reason::kMissingTripId:
      return "missing_trip_id";
    case vehicle_prediction_rejection_reason::kUnresolvedTrip:
      return "unresolved_trip";
    case vehicle_prediction_rejection_reason::kUnscheduledTrip:
      return "unscheduled_trip";
    case vehicle_prediction_rejection_reason::kUnsupportedTripRelationship:
      return "unsupported_trip_relationship";
    case vehicle_prediction_rejection_reason::kInsufficientHistory:
      return "insufficient_history";
    case vehicle_prediction_rejection_reason::kStaleHistory:
      return "stale_history";
    case vehicle_prediction_rejection_reason::kMissingShape:
      return "missing_shape";
    case vehicle_prediction_rejection_reason::kOffShape: return "off_shape";
    case vehicle_prediction_rejection_reason::kAmbiguousProgress:
      return "ambiguous_progress";
    case vehicle_prediction_rejection_reason::kNonMonotonicProgress:
      return "non_monotonic_progress";
    case vehicle_prediction_rejection_reason::kInvalidObservationTime:
      return "invalid_observation_time";
    case vehicle_prediction_rejection_reason::kImpossibleSpeed:
      return "impossible_speed";
    case vehicle_prediction_rejection_reason::kImpossibleTravelTime:
      return "impossible_travel_time";
    case vehicle_prediction_rejection_reason::kTerminal: return "terminal";
  }
  std::unreachable();
}

std::size_t vehicle_prediction_batch::estimated_memory_bytes() const {
  return sizeof(*this) +
         predictions_.capacity() * sizeof(vehicle_stop_prediction) +
         observed_passages_.capacity() * sizeof(observed_stop_passage);
}

vehicle_prediction_engine::vehicle_prediction_engine(
    n::shapes_storage const& shapes, vehicle_prediction_policy policy)
    : impl_{std::make_unique<impl>(shapes, std::move(policy))} {}

vehicle_prediction_engine::~vehicle_prediction_engine() = default;
vehicle_prediction_engine::vehicle_prediction_engine(
    vehicle_prediction_engine&&) noexcept = default;
vehicle_prediction_engine& vehicle_prediction_engine::operator=(
    vehicle_prediction_engine&&) noexcept = default;

vehicle_prediction_batch vehicle_prediction_engine::evaluate(
    n::rt::frun const& run,
    std::span<vehicle_observation const> observations,
    std::int64_t const now) {
  auto result = vehicle_prediction_batch{};
  auto reject = [&](vehicle_prediction_rejection_reason const reason) {
    result.diagnostics_.rejection_ = reason;
    return result;
  };
  if (!run.is_scheduled()) {
    return reject(vehicle_prediction_rejection_reason::kUnscheduledTrip);
  }
  result.transport_ = run.t_;
  auto const timeline = impl_->projector_.stop_timeline(run);
  if (!timeline.has_value()) {
    return reject(vehicle_prediction_rejection_reason::kMissingShape);
  }

  auto fresh = std::vector<vehicle_observation const*>{};
  fresh.reserve(observations.size());
  auto const cutoff = now - impl_->policy_.max_observation_age_seconds_;
  for (auto const& observation : observations) {
    if (observation_time(observation) >= cutoff &&
        observation_time(observation) <= now) {
      fresh.push_back(&observation);
    }
  }
  result.diagnostics_.fresh_observation_count_ = fresh.size();
  if (fresh.empty() && !observations.empty()) {
    return reject(vehicle_prediction_rejection_reason::kStaleHistory);
  }
  if (fresh.size() < impl_->policy_.min_observations_) {
    return reject(vehicle_prediction_rejection_reason::kInsufficientHistory);
  }

  std::ranges::sort(fresh, [](auto const* a, auto const* b) {
    return std::pair{observation_time(*a), a->ingested_time_} <
           std::pair{observation_time(*b), b->ingested_time_};
  });
  auto samples = std::vector<timed_progress>{};
  samples.reserve(fresh.size());
  auto prior = std::optional<trip_progress>{};
  for (auto const* observation : fresh) {
    auto const timestamp = observation_time(*observation);
    if (!samples.empty() && timestamp <= samples.back().time_) {
      continue;
    }
    auto const projected = impl_->projector_.project(
        run, geo::latlng{observation->latitude_, observation->longitude_},
        prior, constraint_for(*observation));
    if (!projected.progress_.has_value()) {
      return reject(rejection_for(projected.status_));
    }
    prior = projected.progress_;
    samples.push_back(
        {.time_ = timestamp,
         .progress_ = *projected.progress_,
         .reported_speed_mps_ = observation->speed_mps_,
         .stopped_at_ = observation->current_status_ == "STOPPED_AT"});
  }
  if (samples.size() < impl_->policy_.min_observations_) {
    return reject(vehicle_prediction_rejection_reason::kInvalidObservationTime);
  }

  auto const velocity = robust_velocity(samples);
  auto const& latest = samples.back();
  auto const stop_idx = next_stop_index(*timeline, latest.progress_);
  if (stop_idx >= timeline->size()) {
    return reject(vehicle_prediction_rejection_reason::kTerminal);
  }
  auto const& anchor_stop = (*timeline)[stop_idx];
  auto const distance_to_anchor =
      std::max(0.0, anchor_stop.distance_along_shape_m_ -
                        latest.progress_.distance_along_shape_m_);
  if (stop_idx + 1U == timeline->size() && distance_to_anchor <= 25.0) {
    return reject(vehicle_prediction_rejection_reason::kTerminal);
  }

  auto predicted_anchor = now;
  auto used_velocity = 0.0;
  if (!(latest.stopped_at_ && distance_to_anchor <= 25.0)) {
    if (!velocity.has_value() || !std::isfinite(*velocity) ||
        *velocity < impl_->policy_.min_progress_velocity_mps_ ||
        *velocity > impl_->policy_.max_progress_velocity_mps_) {
      return reject(vehicle_prediction_rejection_reason::kImpossibleSpeed);
    }
    used_velocity = *velocity;
    auto const travel =
        static_cast<std::int64_t>(std::llround(distance_to_anchor / *velocity));
    if (travel < 0 || travel > impl_->policy_.max_predicted_travel_seconds_) {
      return reject(vehicle_prediction_rejection_reason::kImpossibleTravelTime);
    }
    predicted_anchor += travel;
  }
  auto const anchor_schedule = latest.stopped_at_
                                   ? anchor_stop.scheduled_departure_time_
                                   : anchor_stop.scheduled_arrival_time_;
  auto const delay = predicted_anchor - anchor_schedule;
  result.delay_anchor_static_stop_sequence_ = anchor_stop.static_stop_sequence_;
  result.delay_anchor_seconds_ = delay;
  result.candidate_reference_timestamp_seconds_ = latest.time_;
  result.implied_progress_m_ = latest.progress_.distance_along_shape_m_;
  result.predictions_.reserve(timeline->size() - stop_idx);
  for (auto i = stop_idx; i != timeline->size(); ++i) {
    auto const& stop = (*timeline)[i];
    auto const scheduled = stop.scheduled_arrival_time_;
    auto const predicted = scheduled + delay;
    result.predictions_.push_back(
        {.static_stop_sequence_ = stop.static_stop_sequence_,
         .scheduled_timestamp_seconds_ = scheduled,
         .predicted_timestamp_seconds_ = predicted,
         .delay_seconds_ = delay,
         .horizon_seconds_ = predicted - now});
  }

  auto const corroborating_speed = reported_speed(samples);
  auto const speed_conflict =
      velocity.has_value() && corroborating_speed.has_value() &&
      std::abs(*corroborating_speed - *velocity) > 2.0 &&
      std::abs(*corroborating_speed - *velocity) >
          std::max(*corroborating_speed, *velocity) * 0.5;
  auto const lateral = latest.progress_.lateral_error_m_;
  auto confidence = std::clamp(1.0 - lateral / 100.0, 0.0, 1.0);
  confidence *= std::min(1.0, static_cast<double>(samples.size()) / 4.0);
  if (speed_conflict) {
    confidence *= 0.5;
  }
  result.confidence_ =
      vehicle_prediction_confidence{.score_ = confidence,
                                    .observation_count_ = samples.size(),
                                    .lateral_error_m_ = lateral,
                                    .progress_velocity_mps_ = used_velocity,
                                    .reported_speed_conflict_ = speed_conflict};
  result.observed_passages_ =
      derive_passages(samples, *timeline, impl_->policy_,
                      result.diagnostics_.uncertain_passage_count_);
  return result;
}

}  // namespace motis
