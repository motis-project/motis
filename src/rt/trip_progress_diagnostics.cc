#include "motis/rt/trip_progress_diagnostics.h"

#include <algorithm>
#include <string_view>

#include "geo/latlng.h"

#include "nigiri/rt/frun.h"
#include "nigiri/shapes_storage.h"

#include "motis/config.h"
#include "motis/rt/trip_progress_projection.h"
#include "motis/rt/vehicle_matching.h"
#include "motis/rt/vehicle_observation_history.h"
#include "motis/rt/vehicle_position.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace motis {

namespace {

std::string_view feed_tag(std::string_view const feed) {
  auto const separator = feed.find(':');
  return separator == std::string_view::npos ? feed : feed.substr(0, separator);
}

bool scheduled_relationship_is_stable(
    std::optional<std::string> const& relationship) {
  return !relationship.has_value() || *relationship == "SCHEDULED";
}

vehicle_key key_for(vehicle_positions::vehicle_position const& vehicle) {
  return vehicle.vehicle_.id_.has_value()
             ? vehicle_key{vehicle.feed_id_, *vehicle.vehicle_.id_,
                           vehicle_key_source::kVehicleDescriptor}
             : vehicle_key{vehicle.feed_id_, vehicle.entity_id_,
                           vehicle_key_source::kEntityId};
}

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

bool feed_enabled(config const& c, std::string_view const feed) {
  for (auto i = 0U; i != n::kNumClasses; ++i) {
    if (c.vehicle_eta_mode(feed, static_cast<n::clasz>(i)) !=
        config::timetable::vehicle_eta::mode::off) {
      return true;
    }
  }
  return false;
}

trip_progress_diagnostic_status to_diagnostic_status(
    trip_progress_projection_status const status) {
  switch (status) {
    case trip_progress_projection_status::kProjected:
      return trip_progress_diagnostic_status::kProjected;
    case trip_progress_projection_status::kMissingShape:
      return trip_progress_diagnostic_status::kMissingShape;
    case trip_progress_projection_status::kOffShape:
      return trip_progress_diagnostic_status::kOffShape;
    case trip_progress_projection_status::kAmbiguous:
      return trip_progress_diagnostic_status::kAmbiguous;
    case trip_progress_projection_status::kImplausible:
      return trip_progress_diagnostic_status::kImplausible;
  }
  std::unreachable();
}

}  // namespace

char const* to_str(trip_progress_diagnostic_status const status) {
  switch (status) {
    case trip_progress_diagnostic_status::kProjected: return "projected";
    case trip_progress_diagnostic_status::kStale: return "stale";
    case trip_progress_diagnostic_status::kMissingTripId:
      return "missing_trip_id";
    case trip_progress_diagnostic_status::kUnsupportedScheduleRelationship:
      return "unsupported_schedule_relationship";
    case trip_progress_diagnostic_status::kUnresolvedTrip:
      return "unresolved_trip";
    case trip_progress_diagnostic_status::kUnscheduledTrip:
      return "unscheduled_trip";
    case trip_progress_diagnostic_status::kMissingHistory:
      return "missing_history";
    case trip_progress_diagnostic_status::kMissingShape: return "missing_shape";
    case trip_progress_diagnostic_status::kOffShape: return "off_shape";
    case trip_progress_diagnostic_status::kAmbiguous: return "ambiguous";
    case trip_progress_diagnostic_status::kImplausible: return "implausible";
  }
  std::unreachable();
}

std::vector<trip_progress_diagnostic> evaluate_trip_progress_diagnostics(
    config const& c,
    tag_lookup const& tags,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    n::shapes_storage const* shapes,
    vehicle_positions::vehicle_position_store const& positions,
    vehicle_observation_history const& history,
    std::int64_t const now) {
  auto diagnostics = std::vector<trip_progress_diagnostic>{};
  if (!c.timetable_ || !c.timetable_->vehicle_eta_) {
    return diagnostics;
  }

  auto projector = shapes == nullptr ? std::optional<trip_progress_projector>{}
                                     : std::optional<trip_progress_projector>{
                                           std::in_place, *shapes};
  auto const max_age = c.timetable_->vehicle_eta_->history_.max_age_seconds_;
  auto const cutoff = vehicle_matching::freshness_cutoff(now, max_age);

  for (auto const& vehicle : positions.all()) {
    auto diagnostic = trip_progress_diagnostic{
        .feed_ = std::string{feed_tag(vehicle.feed_id_)}};
    if (!feed_enabled(c, diagnostic.feed_)) {
      continue;
    }
    auto reject = [&](trip_progress_diagnostic_status const status) {
      diagnostic.status_ = status;
      diagnostics.emplace_back(std::move(diagnostic));
    };

    if (!vehicle.trip_.trip_id_.has_value()) {
      reject(trip_progress_diagnostic_status::kMissingTripId);
      continue;
    }
    if (!scheduled_relationship_is_stable(
            vehicle.trip_.schedule_relationship_)) {
      reject(trip_progress_diagnostic_status::kUnsupportedScheduleRelationship);
      continue;
    }
    if (!vehicle_matching::is_fresh(vehicle, cutoff)) {
      reject(trip_progress_diagnostic_status::kStale);
      continue;
    }

    auto run = vehicle_matching::resolve_run(tags, tt, rtt, vehicle);
    if (!run.has_value()) {
      reject(trip_progress_diagnostic_status::kUnresolvedTrip);
      continue;
    }
    if (!run->is_scheduled()) {
      reject(trip_progress_diagnostic_status::kUnscheduledTrip);
      continue;
    }
    diagnostic.mode_ = (*run)[0].get_clasz(n::event_type::kDep);
    if (c.vehicle_eta_mode(diagnostic.feed_, *diagnostic.mode_) ==
        config::timetable::vehicle_eta::mode::off) {
      continue;
    }
    if (!projector.has_value()) {
      reject(trip_progress_diagnostic_status::kMissingShape);
      continue;
    }

    auto const observations = history.observations(key_for(vehicle));
    if (observations.empty()) {
      reject(trip_progress_diagnostic_status::kMissingHistory);
      continue;
    }
    auto prior = std::optional<trip_progress>{};
    auto latest = trip_progress_projection{};
    auto evaluated = false;
    for (auto const& observation : observations) {
      if (observation_time(observation) < cutoff) {
        continue;
      }
      latest = projector->project(
          *run, geo::latlng{observation.latitude_, observation.longitude_},
          prior, constraint_for(observation));
      evaluated = true;
      if (latest.progress_.has_value()) {
        prior = latest.progress_;
      }
    }
    if (!evaluated) {
      reject(trip_progress_diagnostic_status::kMissingHistory);
      continue;
    }
    diagnostic.status_ = to_diagnostic_status(latest.status_);
    if (latest.progress_.has_value()) {
      diagnostic.lateral_error_m_ = latest.progress_->lateral_error_m_;
    }
    diagnostics.emplace_back(std::move(diagnostic));
  }
  return diagnostics;
}

}  // namespace motis
