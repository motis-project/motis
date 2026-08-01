#include "motis/rt/vehicle_prediction_selection.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/logging.h"

#include "motis/config.h"

namespace motis {

namespace {

std::optional<scheduled_provider_stop> find_scheduled_stop(
    resolved_provider_trip const& resolved, unsigned const sequence) {
  auto const it =
      std::ranges::find(resolved.stops_, sequence,
                        &scheduled_provider_stop::static_stop_sequence_);
  return it == end(resolved.stops_) ? std::nullopt : std::optional{*it};
}

std::optional<std::int64_t> event_timestamp(
    transit_realtime::TripUpdate_StopTimeEvent const& event,
    std::optional<std::int64_t> const scheduled) {
  if (event.has_time()) {
    return event.time();
  }
  if (event.has_delay() && scheduled.has_value()) {
    return *scheduled + event.delay();
  }
  return std::nullopt;
}

bool has_timing(provider_stop_timing const& stop) {
  return stop.arrival_timestamp_seconds_.has_value() ||
         stop.departure_timestamp_seconds_.has_value();
}

std::optional<timing_candidate_rejection_reason> validate_candidate(
    timing_source_candidate const& candidate,
    double const minimum_confidence,
    std::int64_t const now,
    vehicle_prediction_selector_policy const& policy) {
  if (candidate.predictions_.empty() || !candidate.physically_reachable_) {
    return timing_candidate_rejection_reason::kPhysicallyUnreachable;
  }
  if (candidate.reference_timestamp_seconds_ > now ||
      now - candidate.reference_timestamp_seconds_ >
          policy.max_candidate_age_seconds_) {
    return timing_candidate_rejection_reason::kStale;
  }
  if (candidate.confidence_ < minimum_confidence) {
    return timing_candidate_rejection_reason::kLowConfidence;
  }
  return std::nullopt;
}

timing_source_candidate const* candidate_for(
    vehicle_prediction_source const source,
    vehicle_prediction_selection_input const& input) {
  switch (source) {
    case vehicle_prediction_source::kProvider:
      return input.provider_ ? &*input.provider_ : nullptr;
    case vehicle_prediction_source::kGps:
      return input.gps_ ? &*input.gps_ : nullptr;
    case vehicle_prediction_source::kSchedule: return nullptr;
  }
  return nullptr;
}

bool explicitly_inconsistent(vehicle_prediction_selection const& selection) {
  return selection.diagnostics_.provider_rejection_ ==
             timing_candidate_rejection_reason::kProgressInconsistent ||
         selection.diagnostics_.provider_rejection_ ==
             timing_candidate_rejection_reason::kTimestampNotComparable ||
         selection.diagnostics_.provider_rejection_ ==
             timing_candidate_rejection_reason::kProgressNotComparable ||
         selection.diagnostics_.provider_rejection_ ==
             timing_candidate_rejection_reason::kPhysicallyUnreachable;
}

}  // namespace

provider_timing_extraction extract_provider_timing(
    transit_realtime::FeedMessage const& message,
    provider_trip_resolver const& resolve) {
  auto result = provider_timing_extraction{};
  for (auto const& entity : message.entity()) {
    if (!entity.has_trip_update()) {
      continue;
    }
    auto const& update = entity.trip_update();
    auto const resolved = resolve(update.trip());
    if (!resolved.has_value() ||
        resolved->transport_ == nigiri::transport::invalid()) {
      ++result.unresolved_trip_updates_;
      continue;
    }

    auto candidate = provider_timing_candidate{
        .transport_ = resolved->transport_,
        .feed_timestamp_seconds_ =
            message.has_header() && message.header().has_timestamp()
                ? std::optional<std::int64_t>{message.header().timestamp()}
                : std::nullopt,
        .trip_update_timestamp_seconds_ =
            update.has_timestamp()
                ? std::optional<std::int64_t>{update.timestamp()}
                : std::nullopt};
    for (auto const& source_stop : update.stop_time_update()) {
      if (!source_stop.has_stop_sequence() ||
          (source_stop.has_schedule_relationship() &&
           source_stop.schedule_relationship() ==
               transit_realtime::TripUpdate_StopTimeUpdate::NO_DATA)) {
        continue;
      }
      auto const sequence = source_stop.stop_sequence();
      auto const scheduled = find_scheduled_stop(*resolved, sequence);
      auto timing = provider_stop_timing{.static_stop_sequence_ = sequence};
      if (source_stop.has_arrival()) {
        timing.arrival_timestamp_seconds_ = event_timestamp(
            source_stop.arrival(),
            scheduled ? scheduled->arrival_timestamp_seconds_ : std::nullopt);
        if (source_stop.arrival().has_delay()) {
          timing.arrival_delay_seconds_ = source_stop.arrival().delay();
        }
      }
      if (source_stop.has_departure()) {
        timing.departure_timestamp_seconds_ = event_timestamp(
            source_stop.departure(),
            scheduled ? scheduled->departure_timestamp_seconds_ : std::nullopt);
        if (source_stop.departure().has_delay()) {
          timing.departure_delay_seconds_ = source_stop.departure().delay();
        }
      }
      if (has_timing(timing)) {
        candidate.stops_.emplace_back(std::move(timing));
      }
    }
    if (candidate.stops_.empty()) {
      ++result.operational_only_trip_updates_;
    } else {
      result.candidates_.emplace_back(std::move(candidate));
    }
  }
  return result;
}

char const* to_str(vehicle_prediction_source const source) {
  switch (source) {
    case vehicle_prediction_source::kProvider: return "provider";
    case vehicle_prediction_source::kGps: return "gps";
    case vehicle_prediction_source::kSchedule: return "schedule";
  }
  std::unreachable();
}

char const* to_str(vehicle_prediction_selection_reason const reason) {
  switch (reason) {
    case vehicle_prediction_selection_reason::kProviderOnly:
      return "provider_only";
    case vehicle_prediction_selection_reason::kGpsOnly: return "gps_only";
    case vehicle_prediction_selection_reason::kProviderHigherConfidence:
      return "provider_higher_confidence";
    case vehicle_prediction_selection_reason::kGpsHigherConfidence:
      return "gps_higher_confidence";
    case vehicle_prediction_selection_reason::kProviderProgressInconsistent:
      return "provider_progress_inconsistent";
    case vehicle_prediction_selection_reason::kProviderRecoveryPending:
      return "provider_recovery_pending";
    case vehicle_prediction_selection_reason::kProviderRecovered:
      return "provider_recovered";
    case vehicle_prediction_selection_reason::kSourceHysteresis:
      return "source_hysteresis";
    case vehicle_prediction_selection_reason::kNoUsableCandidate:
      return "no_usable_candidate";
    case vehicle_prediction_selection_reason::kPolicyUnavailable:
      return "policy_unavailable";
  }
  std::unreachable();
}

char const* to_str(timing_candidate_rejection_reason const reason) {
  switch (reason) {
    case timing_candidate_rejection_reason::kStale: return "stale";
    case timing_candidate_rejection_reason::kLowConfidence:
      return "low_confidence";
    case timing_candidate_rejection_reason::kPhysicallyUnreachable:
      return "physically_unreachable";
    case timing_candidate_rejection_reason::kTimestampNotComparable:
      return "timestamp_not_comparable";
    case timing_candidate_rejection_reason::kProgressNotComparable:
      return "progress_not_comparable";
    case timing_candidate_rejection_reason::kProgressInconsistent:
      return "progress_inconsistent";
  }
  std::unreachable();
}

bool vehicle_prediction_selector_policy::valid() const {
  return max_candidate_age_seconds_ > 0 && max_timestamp_skew_seconds_ >= 0 &&
         max_progress_difference_m_ >= 0.0 && min_provider_confidence_ >= 0.0 &&
         min_provider_confidence_ <= 1.0 && min_gps_confidence_ >= 0.0 &&
         min_gps_confidence_ <= 1.0 &&
         min_source_switch_confidence_advantage_ >= 0.0 &&
         state_ttl_seconds_ > 0 && flap_window_seconds_ >= 0 &&
         minute_boundary_hysteresis_seconds_ >= 0;
}

vehicle_prediction_selection select_vehicle_prediction_source(
    vehicle_prediction_selection_input const& input,
    vehicle_prediction_selector_policy const& policy) {
  auto result = vehicle_prediction_selection{};
  if (!policy.valid()) {
    result.reason_ = vehicle_prediction_selection_reason::kPolicyUnavailable;
    return result;
  }

  if (input.provider_) {
    result.diagnostics_.provider_rejection_ =
        validate_candidate(*input.provider_, policy.min_provider_confidence_,
                           input.now_seconds_, policy);
  }
  if (input.gps_) {
    result.diagnostics_.gps_rejection_ = validate_candidate(
        *input.gps_, policy.min_gps_confidence_, input.now_seconds_, policy);
  }
  auto provider_ok = input.provider_.has_value() &&
                     !result.diagnostics_.provider_rejection_.has_value();
  auto gps_ok =
      input.gps_.has_value() && !result.diagnostics_.gps_rejection_.has_value();

  if (provider_ok && gps_ok) {
    auto const skew = std::llabs(input.provider_->reference_timestamp_seconds_ -
                                 input.gps_->reference_timestamp_seconds_);
    result.diagnostics_.candidate_timestamp_skew_seconds_ = skew;
    if (skew > policy.max_timestamp_skew_seconds_) {
      result.diagnostics_.provider_rejection_ =
          timing_candidate_rejection_reason::kTimestampNotComparable;
      provider_ok = false;
    } else if (!input.provider_->implied_progress_m_.has_value() ||
               !input.gps_->implied_progress_m_.has_value()) {
      result.diagnostics_.provider_rejection_ =
          timing_candidate_rejection_reason::kProgressNotComparable;
      provider_ok = false;
    } else {
      auto const difference = std::abs(*input.provider_->implied_progress_m_ -
                                       *input.gps_->implied_progress_m_);
      result.diagnostics_.progress_difference_m_ = difference;
      if (difference > policy.max_progress_difference_m_) {
        result.diagnostics_.provider_rejection_ =
            timing_candidate_rejection_reason::kProgressInconsistent;
        provider_ok = false;
      }
    }
  }

  auto choose = [&](timing_source_candidate const& candidate,
                    vehicle_prediction_selection_reason const reason) {
    result.source_ = candidate.source_;
    result.reason_ = reason;
    result.predictions_ = candidate.predictions_;
    result.diagnostics_.selected_confidence_ = candidate.confidence_;
    for (auto const& prediction : candidate.predictions_) {
      result.diagnostics_.rendered_delays_.push_back(
          {.static_stop_sequence_ = prediction.static_stop_sequence_,
           .raw_delay_seconds_ = prediction.delay_seconds_,
           .rendered_delay_minutes_ =
               rendered_delay_minutes(prediction.delay_seconds_)});
    }
  };
  if (provider_ok && gps_ok) {
    if (input.gps_->confidence_ > input.provider_->confidence_) {
      choose(*input.gps_,
             vehicle_prediction_selection_reason::kGpsHigherConfidence);
    } else {
      choose(*input.provider_,
             vehicle_prediction_selection_reason::kProviderHigherConfidence);
    }
  } else if (provider_ok) {
    choose(*input.provider_,
           vehicle_prediction_selection_reason::kProviderOnly);
  } else if (gps_ok) {
    choose(
        *input.gps_,
        result.diagnostics_.provider_rejection_ ==
                timing_candidate_rejection_reason::kProgressInconsistent
            ? vehicle_prediction_selection_reason::kProviderProgressInconsistent
            : vehicle_prediction_selection_reason::kGpsOnly);
  }
  return result;
}

vehicle_prediction_selection vehicle_prediction_selection_state::select(
    vehicle_prediction_selection_input const& input,
    vehicle_prediction_selector_policy const& policy,
    bool const completed) {
  expire(input.now_seconds_, policy);
  auto const existing =
      std::ranges::find(entries_, input.transport_, &entry::transport_);
  if (completed) {
    if (existing != end(entries_)) {
      entries_.erase(existing);
    }
    return select_vehicle_prediction_source(input, policy);
  }

  auto selection = select_vehicle_prediction_source(input, policy);
  if (!policy.valid() || input.transport_ == nigiri::transport::invalid()) {
    return selection;
  }
  auto& state = existing == end(entries_)
                    ? entries_.emplace_back(
                          entry{.transport_ = input.transport_,
                                .selected_ = selection.source_,
                                .last_seen_seconds_ = input.now_seconds_})
                    : *existing;
  state.last_seen_seconds_ = input.now_seconds_;
  auto const prior_source = state.selected_;

  if (explicitly_inconsistent(selection)) {
    state.provider_rejected_ = true;
    state.provider_consistent_cycles_ = 0U;
  } else if (state.provider_rejected_ && input.provider_.has_value() &&
             !selection.diagnostics_.provider_rejection_.has_value()) {
    ++state.provider_consistent_cycles_;
    if (state.provider_consistent_cycles_ < 2U &&
        selection.source_ == vehicle_prediction_source::kProvider) {
      if (auto const* const gps =
              candidate_for(vehicle_prediction_source::kGps, input)) {
        selection.source_ = vehicle_prediction_source::kGps;
        selection.predictions_ = gps->predictions_;
        selection.diagnostics_.selected_confidence_ = gps->confidence_;
        selection.diagnostics_.rendered_delays_.clear();
        for (auto const& prediction : gps->predictions_) {
          selection.diagnostics_.rendered_delays_.push_back(
              {.static_stop_sequence_ = prediction.static_stop_sequence_,
               .raw_delay_seconds_ = prediction.delay_seconds_,
               .rendered_delay_minutes_ =
                   rendered_delay_minutes(prediction.delay_seconds_)});
        }
      } else {
        selection.source_ = vehicle_prediction_source::kSchedule;
        selection.predictions_.clear();
        selection.diagnostics_.selected_confidence_.reset();
        selection.diagnostics_.rendered_delays_.clear();
      }
      selection.reason_ =
          vehicle_prediction_selection_reason::kProviderRecoveryPending;
    } else if (state.provider_consistent_cycles_ >= 2U) {
      state.provider_rejected_ = false;
      if (selection.source_ == vehicle_prediction_source::kProvider) {
        selection.reason_ =
            vehicle_prediction_selection_reason::kProviderRecovered;
        selection.diagnostics_.provider_recovery_ = true;
      }
    }
  }

  if (selection.source_ != prior_source &&
      !explicitly_inconsistent(selection) &&
      !selection.diagnostics_.provider_recovery_) {
    auto const* previous = candidate_for(prior_source, input);
    auto const* selected = candidate_for(selection.source_, input);
    auto const previous_rejected =
        prior_source == vehicle_prediction_source::kProvider
            ? selection.diagnostics_.provider_rejection_.has_value()
        : prior_source == vehicle_prediction_source::kGps
            ? selection.diagnostics_.gps_rejection_.has_value()
            : true;
    if (previous != nullptr && selected != nullptr && !previous_rejected &&
        selected->confidence_ - previous->confidence_ <
            policy.min_source_switch_confidence_advantage_) {
      selection.source_ = prior_source;
      selection.predictions_ = previous->predictions_;
      selection.diagnostics_.selected_confidence_ = previous->confidence_;
      selection.reason_ =
          vehicle_prediction_selection_reason::kSourceHysteresis;
      selection.diagnostics_.rendered_delays_.clear();
      for (auto const& prediction : previous->predictions_) {
        selection.diagnostics_.rendered_delays_.push_back(
            {.static_stop_sequence_ = prediction.static_stop_sequence_,
             .raw_delay_seconds_ = prediction.delay_seconds_,
             .rendered_delay_minutes_ =
                 rendered_delay_minutes(prediction.delay_seconds_)});
      }
    }
  }

  if (selection.source_ != prior_source) {
    selection.diagnostics_.source_transition_ = true;
    if (state.last_transition_seconds_.has_value() &&
        input.now_seconds_ - *state.last_transition_seconds_ <=
            policy.flap_window_seconds_) {
      selection.diagnostics_.flap_ = true;
    }
    state.last_transition_seconds_ = input.now_seconds_;
    state.selected_ = selection.source_;
  }
  selection.diagnostics_.provider_consistent_cycles_ =
      state.provider_consistent_cycles_;
  for (auto& rendered : selection.diagnostics_.rendered_delays_) {
    auto const prior =
        std::ranges::find_if(state.rendered_minutes_, [&](auto const& x) {
          return x.first == rendered.static_stop_sequence_;
        });
    auto const previous_minutes =
        prior == end(state.rendered_minutes_)
            ? std::optional<std::int64_t>{}
            : std::optional<std::int64_t>{prior->second};
    rendered.rendered_delay_minutes_ = rendered_delay_minutes_with_hysteresis(
        rendered.raw_delay_seconds_, previous_minutes,
        policy.minute_boundary_hysteresis_seconds_);
    if (prior == end(state.rendered_minutes_)) {
      state.rendered_minutes_.emplace_back(rendered.static_stop_sequence_,
                                           rendered.rendered_delay_minutes_);
    } else {
      prior->second = rendered.rendered_delay_minutes_;
    }
  }
  return selection;
}

void vehicle_prediction_selection_state::expire(
    std::int64_t const now_seconds,
    vehicle_prediction_selector_policy const& policy) {
  if (!policy.valid()) {
    entries_.clear();
    return;
  }
  std::erase_if(entries_, [&](entry const& x) {
    return now_seconds < x.last_seen_seconds_ ||
           now_seconds - x.last_seen_seconds_ > policy.state_ttl_seconds_;
  });
}

std::int64_t rendered_delay_minutes(std::int64_t const raw_seconds) {
  return raw_seconds / 60;
}

std::int64_t rendered_delay_minutes_with_hysteresis(
    std::int64_t const raw_seconds,
    std::optional<std::int64_t> const previous,
    std::int64_t const hysteresis) {
  auto const candidate = rendered_delay_minutes(raw_seconds);
  if (!previous.has_value() || candidate == *previous || hysteresis <= 0 ||
      std::llabs(candidate - *previous) > 1) {
    return candidate;
  }
  if (candidate > *previous) {
    auto const boundary = candidate > 0 ? candidate * 60 : (*previous) * 60;
    return raw_seconds >= boundary + hysteresis ? candidate : *previous;
  }
  auto const boundary = candidate < 0 ? candidate * 60 : (*previous) * 60;
  return raw_seconds <= boundary - hysteresis ? candidate : *previous;
}

double vehicle_prediction_shadow_summary::flap_rate() const {
  return evaluated_ == 0U
             ? 0.0
             : static_cast<double>(flaps_) / static_cast<double>(evaluated_);
}

vehicle_prediction_shadow_cycle_result evaluate_vehicle_prediction_shadow_cycle(
    config const& c,
    std::span<vehicle_prediction_shadow_request const> const requests,
    vehicle_prediction_selector_policy_resolver const& resolve_policy,
    vehicle_prediction_selection_state& state) {
  auto result = vehicle_prediction_shadow_cycle_result{};
  result.selections_.reserve(requests.size());
  auto summary_for = [&](vehicle_prediction_shadow_request const& request)
      -> vehicle_prediction_shadow_summary& {
    auto const it = std::ranges::find_if(
        result.summaries_, [&](vehicle_prediction_shadow_summary const& x) {
          return x.feed_ == request.feed_ && x.mode_ == request.mode_;
        });
    return it == end(result.summaries_)
               ? result.summaries_.emplace_back(
                     vehicle_prediction_shadow_summary{.feed_ = request.feed_,
                                                       .mode_ = request.mode_})
               : *it;
  };

  for (auto const& request : requests) {
    if (c.vehicle_eta_mode(request.feed_, request.mode_) ==
        config::timetable::vehicle_eta::mode::off) {
      continue;
    }
    auto const policy = resolve_policy(request.feed_, request.mode_);
    auto selection =
        policy.has_value()
            ? state.select(request.selection_, *policy, request.completed_)
            : vehicle_prediction_selection{
                  .reason_ =
                      vehicle_prediction_selection_reason::kPolicyUnavailable};
    auto& summary = summary_for(request);
    ++summary.evaluated_;
    switch (selection.source_) {
      case vehicle_prediction_source::kProvider:
        ++summary.provider_selected_;
        break;
      case vehicle_prediction_source::kGps: ++summary.gps_selected_; break;
      case vehicle_prediction_source::kSchedule:
        ++summary.schedule_fallback_;
        break;
    }
    summary.transitions_ += selection.diagnostics_.source_transition_ ? 1U : 0U;
    summary.provider_recoveries_ +=
        selection.diagnostics_.provider_recovery_ ? 1U : 0U;
    summary.rejections_ +=
        selection.diagnostics_.provider_rejection_.has_value() ? 1U : 0U;
    summary.rejections_ +=
        selection.diagnostics_.gps_rejection_.has_value() ? 1U : 0U;
    summary.flaps_ += selection.diagnostics_.flap_ ? 1U : 0U;
    result.selections_.emplace_back(std::move(selection));
  }
  return result;
}

void log_vehicle_prediction_shadow_cycle(
    std::span<vehicle_prediction_shadow_summary const> const summaries) {
  for (auto const& summary : summaries) {
    nigiri::log(
        nigiri::log_lvl::info, "motis.vehicle_eta",
        "shadow cycle: feed={}, mode={}, evaluated={}, provider={}, gps={}, "
        "fallback={}, transitions={}, recovery={}, rejections={}, flaps={}, "
        "flap_rate={}",
        summary.feed_, nigiri::to_str(summary.mode_), summary.evaluated_,
        summary.provider_selected_, summary.gps_selected_,
        summary.schedule_fallback_, summary.transitions_,
        summary.provider_recoveries_, summary.rejections_, summary.flaps_,
        summary.flap_rate());
  }
}

}  // namespace motis
